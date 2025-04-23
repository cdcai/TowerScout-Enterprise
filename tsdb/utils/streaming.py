"""
This module contains utilities for extending Spark Structured Streams
"""
from datetime import datetime, timedelta
import threading

from pyspark.sql.streaming import StreamingQuery, StreamingQueryListener
import pyspark.sql.streaming.listener as listener


class StreamShutdownListener(StreamingQueryListener):
    """
    A listener that shuts down a spark stream after a certain number of minutes.

    To use this object:
        listener = StreamShutdownListener(timeout=1)
        spark.streams.addListener(listener)

        stream_job = spark.readStream().format("rate").start()
        listener.set_stream(stream_job)
        stream_job.awaitTermination()

    Args:
        timeout: Number of minutes to shut stream off
    """
    def __init__(self, timeout: int):
        self.max_duration = timedelta(minutes=timeout)
        self._idle_start_time = None
        self._stream = None
        self._lock = threading.Lock()
    
    @staticmethod
    def utc_string_to_datetime(date: str) -> datetime:
        """
        Converts a UTC string to a datetime object.
        """
        return datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ")

    def onQueryStarted(self, event: listener.QueryStartedEvent) -> None:
        pass

    def onQueryIdle(self, event: listener.QueryIdleEvent) -> None:
        """
        Manages the countdown to stream shutdown. If no idle time exists, sets it to the current time.
        Otherwise, compares current time to the set idle time. If the timedelta value is reached, stops the stream.
        """
        current_time = self.utc_string_to_datetime(event.timestamp)
        
        if self._idle_start_time is None:
            with self._lock:
                self._idle_start_time = current_time
        
        if (current_time - self._idle_start_time) > self.max_duration:
            self._stream.stop()

    def onQueryProgress(self, event: listener.QueryProgressEvent) -> None:
        """
        If the query is making progress, we want to reset the idle time so progress
        can continue. This makes the shutdown more graceful.
        """
        if self._idle_start_time is not None:
            with self._lock:
                self._idle_start_time = None
    
    def onQueryTerminated(self, event: listener.QueryTerminatedEvent) -> None:
        pass

    def set_stream(self, stream: StreamingQuery) -> None:
        """
        Adds a query object to the listener that will be shut down when max_duration is reached.
        """
        self._stream = stream
