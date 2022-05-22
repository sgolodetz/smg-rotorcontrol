import asyncio
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import sounddevice
import threading

from queue import Queue
from typing import Callable, Dict, List, Optional, Set, Tuple

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStream, TranscriptResultStreamHandler
from amazon_transcribe.model import Result, TranscriptEvent

from smg.rotory.drones import Drone

from .drone_controller import DroneController


class AWSTranscribeDroneController(DroneController):
    """An AWS Transcribe-based flight controller for a drone."""

    # NESTED TYPES

    class StreamHandler(TranscriptResultStreamHandler):
        """A handler that can handle transcript events from a stream processed by AWS Transcribe."""

        # CONSTRUCTOR

        def __init__(self, stream: TranscriptResultStream, command_queue: "Queue[str]", *, debug: bool = True):
            """
            Construct a handler that can handle transcript events from a stream processed by AWS Transcribe.

            :param stream:          The stream.
            :param command_queue:   The queue onto which to push any recognised drone commands.
            :param debug:           Whether or not to print out debug messages.
            """
            super().__init__(stream)

            self.__command_queue: "Queue[str]" = command_queue
            self.__debug: bool = debug

        # PUBLIC ASYNCHRONOUS METHODS

        async def handle_transcript_event(self, transcript_event: TranscriptEvent) -> None:
            """
            Handle a transcript event from a stream processed by AWS Transcribe.

            :param transcript_event:    The event.
            """
            # Get a list of transcription results, each of which corresponds to a portion of the input audio stream.
            results: List[Result] = transcript_event.transcript.results

            # For each transcription result:
            for result in results:
                # For each possible transcription of the corresponding portion of the input audio stream:
                for alt in result.alternatives:
                    # Get the transcription, convert it to lower case, and then remove any character that is neither
                    # alphanumeric nor whitespace.
                    transcript: str = alt.transcript
                    transcript = transcript.lower()
                    transcript = "".join(filter(lambda c: str.isalnum(c) or str.isspace(c), transcript))

                    # If we're debugging, print out both the transcription and the time it took to compute it.
                    if self.__debug:
                        compute_time: float = round(result.end_time - result.start_time, 3)
                        partiality: str = "partial" if result.is_partial else "full"
                        print(f"Transcription: {transcript} ({compute_time}s; {partiality})")

                    # Add any relevant drone commands that have been found in the audio stream to the command queue.
                    possible_commands: Set[str] = {
                        "back", "backward", "down",  "forward", "land", "level", "move left", "move right",
                        "stop", "straight", "take off", "turn left", "turn right", "up"
                    }

                    for command in possible_commands:
                        if command in transcript:
                            self.__command_queue.put(command)

    # CONSTRUCTOR

    def __init__(self, *, audio_input_device: Optional[int] = None, drone: Drone):
        """
        Construct an AWS Transcribe-based flight controller for a drone.

        :param audio_input_device:  The index of the device to use for audio input.
        :param drone:               The drone.
        """
        super().__init__()

        self.__alive: bool = False

        self.__audio_input_device: Optional[int] = audio_input_device
        self.__command_queue: "Queue[str]" = Queue()
        self.__drone: Drone = drone
        self.__transcription_gather: Optional[asyncio.Future] = None

        # Set the linear and angular rates to use when controlling the drone.
        self.__forward_rate: float = 0.2
        self.__right_rate: float = 0.2
        self.__turn_rate: float = 0.1
        self.__up_rate: float = 0.1

        # Set up and start the transcription thread.
        self.__transcription_thread: threading.Thread = threading.Thread(target=self.__run_transcription)
        self.__transcription_thread.start()

        self.__alive = True

    # PUBLIC METHODS

    def iterate(self, *, altitude: Optional[float] = None, events: Optional[List[pygame.event.Event]] = None,
                image: np.ndarray, image_timestamp: Optional[float] = None,
                intrinsics: Tuple[float, float, float, float], tracker_c_t_i: Optional[np.ndarray] = None) -> None:
        """
        Run an iteration of the controller.

        :param altitude:            The most recent altitude (in m) for the drone, as measured by any height sensor
                                    it is carrying (optional).
        :param events:              An optional list of PyGame events that have happened since the last iteration.
        :param image:               The most recent image from the drone.
        :param image_timestamp:     The timestamp of the most recent image from the drone (optional).
        :param intrinsics:          The intrinsics of the drone's camera.
        :param tracker_c_t_i:       The 6D pose of the drone, expressed as a 4x4 matrix representing a transformation
                                    from initial camera space to current camera space, as estimated by any tracker that
                                    is running (optional). Note that if the tracker is monocular, the transformation is
                                    unlikely to be scale-correct.
        """
        # If there are drone commands on the queue that have not yet been executed:
        while not self.__command_queue.empty():
            # Pop the first command from the queue.
            command: str = self.__command_queue.get()

            # Try to get the drone's current state. Note that we have to do this for each iteration of the loop,
            # since executing commands may affect the state as we go along.
            drone_state: Optional[Drone.EState] = self.__drone.get_state()

            # If the drone's current state isn't available, raise an exception.
            if drone_state is None:
                raise RuntimeError("Error: The AWS Transcribe controller requires the drone's state to be available")

            # Try to run any commands that can be executed in the drone's current state.
            command_runners: Dict[Drone.EState, List[Tuple[List[str], Callable[[], None]]]] = {
                Drone.IDLE: [
                    (["take off"], lambda: self.__drone.takeoff())
                ],
                Drone.FLYING: [
                    (["back", "backward"], lambda: self.__drone.move_forward(-self.__forward_rate)),
                    (["down"], lambda: self.__drone.move_up(-self.__up_rate)),
                    (["forward"], lambda: self.__drone.move_forward(self.__forward_rate)),
                    (["land"], lambda: [self.__drone.stop(), self.__drone.land()]),
                    (["level"], lambda: self.__drone.move_up(0.0)),
                    (["move left"], lambda: self.__drone.move_right(-self.__right_rate)),
                    (["move right"], lambda: self.__drone.move_right(self.__right_rate)),
                    (["stop"], lambda: self.__drone.stop()),
                    (["straight"], lambda: self.__drone.turn(0.0)),
                    (["turn left"], lambda: self.__drone.turn(-self.__turn_rate)),
                    (["turn right"], lambda: self.__drone.turn(self.__turn_rate)),
                    (["up"], lambda: self.__drone.move_up(self.__up_rate))
                ]
            }

            for required_state, command_runners_for_state in command_runners.items():
                if drone_state == required_state:
                    for commands, command_runner in command_runners_for_state:
                        if command in commands:
                            print(f"Command: {command}")
                            command_runner()

    def terminate(self) -> None:
        """Tell the controller to terminate."""
        if self.__alive:
            # Stop any ongoing transcription.
            if self.__transcription_gather is not None:
                self.__transcription_gather.cancel()

            # Join any running threads.
            self.__transcription_thread.join()

            self.__alive = False

    # PRIVATE METHODS

    def __run_transcription(self) -> None:
        """Transcribe audio from the user using AWS Transcribe."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self.__run_transcription_async())
        except:
            # FIXME: Investigate what types of exception (if any) are actually being caught here.
            pass
        finally:
            loop.close()

    # PRIVATE ASYNCHRONOUS METHODS

    async def __mic_stream(self):
        """Asynchronously yield audio chunks from the microphone as they become available."""
        loop = asyncio.get_event_loop()
        input_queue = asyncio.Queue()

        # noinspection PyUnusedLocal
        def callback(indata, frame_count, time_info, status):
            loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

        # Initiate the audio stream, and asynchronously yield the audio chunks as they become available.
        with sounddevice.RawInputStream(
            blocksize=1024 * 2, callback=callback, channels=1, device=self.__audio_input_device,
            dtype="int16", samplerate=16000
        ):
            while True:
                yield await input_queue.get()

    async def __run_transcription_async(self) -> None:
        # Set up the AWS Transcribe streaming client.
        client = TranscribeStreamingClient(region="eu-west-1")

        # Start transcription to generate our async stream
        stream = await client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=16000,
            media_encoding="pcm",
            enable_partial_results_stabilization=True,
            partial_results_stability="high"
        )

        # Instantiate our handler and start processing events
        handler = AWSTranscribeDroneController.StreamHandler(stream.output_stream, self.__command_queue)
        self.__transcription_gather = asyncio.gather(
            self.__write_chunks(stream), handler.handle_events()
        )
        await self.__transcription_gather

    async def __write_chunks(self, stream):
        # This connects the raw audio chunks generator coming from the microphone
        # and passes them along to the transcription stream.
        async for chunk, status in self.__mic_stream():
            await stream.input_stream.send_audio_event(audio_chunk=chunk)
        await stream.input_stream.end_stream()
