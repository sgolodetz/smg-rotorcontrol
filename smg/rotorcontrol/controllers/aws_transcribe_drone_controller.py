import asyncio
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import sounddevice
import threading

from queue import Queue
from typing import Dict, List, Optional, Tuple

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStream, TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

from smg.rotory.drones import Drone

from .drone_controller import DroneController


class AWSTranscribeDroneController(DroneController):
    """An AWS Transcribe-based flight controller for a drone."""

    # NESTED TYPES

    class StreamHandler(TranscriptResultStreamHandler):
        """TODO"""

        # CONSTRUCTOR

        def __init__(self, stream: TranscriptResultStream, command_queue: Queue[str], *, debug: bool = True):
            """
            TODO

            :param stream:          TODO
            :param command_queue:   TODO
            :param debug:           TODO
            """
            super().__init__(stream)

            self.__command_queue: Queue[str] = command_queue
            self.__debug: bool = debug

        # PUBLIC ASYNCHRONOUS METHODS

        async def handle_transcript_event(self, transcript_event: TranscriptEvent) -> None:
            """
            TODO

            :param transcript_event:    TODO
            """
            # TODO
            # This handler can be implemented to handle transcriptions as needed.
            # Here's an example to get started.
            results = transcript_event.transcript.results

            # Only print final result when is_partial is FALSE to get final transcript result without duplicates.
            for result in results:
                for alt in result.alternatives:
                    if result.is_partial:
                        partial_result = alt.transcript
                        partial_result = partial_result.lower()  # Convert result to case-insensitive.
                        partial_result = "".join(filter(lambda c: str.isalnum(c) or str.isspace(c), partial_result))
                        print("Transcription: ", partial_result)
                        # print(result.start_time, result.end_time)
                        compute_time = result.end_time - result.start_time
                        print("Compute time (sec): ", round(compute_time, 3))  # Transcription compute time.

                        voice_to_command: Dict[str, str] = {
                            "back": "back",
                            "backward": "backward",
                            "down": "down",
                            "forward": "forward",
                            "land": "land",
                            "level": "level",
                            "move left": "move left",
                            "move right": "move right",
                            "rotate left": "rotate left",
                            "rotate right": "rotate right",
                            "stop": "stop",
                            "straight": "straight",
                            "take off": "take off",
                            "up": "up"
                        }

                        for voice, command in voice_to_command.items():
                            if voice in partial_result:
                                self.__command_queue.put(command)

    # CONSTRUCTOR

    def __init__(self, *, drone: Drone):
        """
        Construct an AWS Transcribe-based flight controller for a drone.

        :param drone:   The drone.
        """
        super().__init__()

        self.__alive: bool = False

        # TODO: Comment here.
        self.__command_queue: Queue[str] = Queue[str]()
        self.__drone: Drone = drone
        self.__transcription_gather: Optional[asyncio.Future] = None

        # TODO: Comment here.
        self.__forward_rate: float = 0.2
        self.__right_rate: float = 0.2
        self.__turn_rate: float = 0.2
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
            # TODO: Clean this up.
            if drone_state == Drone.IDLE:
                if command == "take off":
                    print(f"Command: {command}")
                    self.__drone.takeoff()
            elif drone_state == Drone.FLYING:
                if command == "back" or command == "backward":
                    print(f"Command: {command}")
                    self.__drone.move_forward(-self.__forward_rate)
                elif command == "down":
                    print(f"Command: {command}")
                    self.__drone.move_up(-self.__up_rate)
                elif command == "forward":
                    print(f"Command: {command}")
                    self.__drone.move_forward(self.__forward_rate)
                elif command == "land":
                    print(f"Command: {command}")
                    self.__drone.stop()
                    self.__drone.land()
                elif command == "level":
                    print(f"Command: {command}")
                    self.__drone.move_up(0.0)
                elif command == "move left":
                    print(f"Command: {command}")
                    self.__drone.move_right(-self.__right_rate)
                elif command == "move right":
                    print(f"Command: {command}")
                    self.__drone.move_right(self.__right_rate)
                elif command == "rotate left":
                    print(f"Command: {command}")
                    self.__drone.turn(-self.__turn_rate)
                elif command == "rotate right":
                    print(f"Command: {command}")
                    self.__drone.turn(self.__turn_rate)
                elif command == "stop":
                    print(f"Command: {command}")
                    self.__drone.stop()
                elif command == "straight":
                    print(f"Command: {command}")
                    self.__drone.turn(0.0)
                elif command == "up":
                    print(f"Command: {command}")
                    self.__drone.move_up(self.__up_rate)

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
        # TODO: Comment this.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.__run_transcription_async())
        except:
            pass
        finally:
            loop.close()

    # PRIVATE ASYNCHRONOUS METHODS

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
            AWSTranscribeDroneController.__write_chunks(stream), handler.handle_events()
        )
        await self.__transcription_gather

    # PRIVATE ASYNCHRONOUS STATIC METHODS

    @staticmethod
    async def __mic_stream():
        # This function wraps the raw input stream from the microphone forwarding
        # the blocks to an asyncio.Queue.
        loop = asyncio.get_event_loop()
        input_queue = asyncio.Queue()

        def callback(indata, frame_count, time_info, status):
            loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

        # Be sure to use the correct parameters for the audio stream that matches
        # the audio formats described for the source language you'll be using:
        # https://docs.aws.amazon.com/transcribe/latest/dg/streaming.html
        stream = sounddevice.RawInputStream(
            device=10,
            channels=1,
            samplerate=16000,
            callback=callback,
            blocksize=1024 * 2,
            dtype="int16",
        )
        # Initiate the audio stream and asynchronously yield the audio chunks
        # as they become available.
        with stream:
            while True:
                indata, status = await input_queue.get()
                yield indata, status

    @staticmethod
    async def __write_chunks(stream):
        # This connects the raw audio chunks generator coming from the microphone
        # and passes them along to the transcription stream.
        async for chunk, status in AWSTranscribeDroneController.__mic_stream():
            await stream.input_stream.send_audio_event(audio_chunk=chunk)
        await stream.input_stream.end_stream()
