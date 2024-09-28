import { intervalToDuration } from 'date-fns';
import { useCallback, useEffect, useRef, useState } from 'react';

export const useCallDuration = () => {
  const interval = useRef<number | null>(null);
  const startTime = useRef<number | null>(null);

  const [timestamp, setTimestamp] = useState<string | null>(null);

  const start = useCallback(() => {
    startTime.current = Date.now();

    setTimestamp('00:00:00');

    interval.current = window.setInterval(() => {
      if (startTime.current) {
        const duration = intervalToDuration({
          start: startTime.current,
          end: Date.now(),
        });

        const hours = (duration.hours ?? 0).toString().padStart(2, '0');
        const minutes = (duration.minutes ?? 0).toString().padStart(2, '0');
        const seconds = (duration.seconds ?? 0).toString().padStart(2, '0');

        setTimestamp(`${hours}:${minutes}:${seconds}`);
      }
    }, 500);
  }, []);

  const stop = useCallback(() => {
    if (interval.current) {
      window.clearInterval(interval.current);
      interval.current = null;
    }
  }, []);

  const reset = useCallback(() => {
    setTimestamp(null);
  }, []);

  useEffect(() => {
    // clean up on unmount
    return () => {
      if (interval.current) {
        window.clearInterval(interval.current);
        interval.current = null;
      }
    };
  }, []);

  return { timestamp, start, stop, reset };
};
