import { useState, useEffect, useRef } from 'react';

export function useThrottle<T>(value: T, interval: number): T {
  const [throttledValue, setThrottledValue] = useState(value);
  const lastUpdated = useRef(0);

  useEffect(() => {
    const now = Date.now();
    const elapsed = now - lastUpdated.current;

    if (elapsed >= interval) {
      lastUpdated.current = now;
      setThrottledValue(value);
      return;
    }

    const timer = setTimeout(() => {
      lastUpdated.current = Date.now();
      setThrottledValue(value);
    }, interval - elapsed);

    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, interval]);

  return throttledValue;
}
