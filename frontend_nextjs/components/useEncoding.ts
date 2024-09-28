// cspell:ignore dataavailable
import { useCallback, useRef, useState } from 'react';

type PermissionStatus = 'prompt' | 'granted' | 'denied';

export const getAudioStream = async (): Promise<MediaStream> => {
  return navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
    video: false,
  });
};

export const checkForAudioTracks = (stream: MediaStream) => {
  const tracks = stream.getAudioTracks();

  if (tracks.length === 0) {
    throw new Error('No audio tracks');
  }
  if (tracks.length > 1) {
    throw new Error('Multiple audio tracks');
  }
  const track = tracks[0];
  if (!track) {
    throw new Error('No audio track');
  }
};

const useEncoding = () => {
  const [permission, setPermission] = useState<PermissionStatus>('prompt');

  const streamRef = useRef<MediaStream | null>(null);

  const getStream = useCallback(async () => {
    try {
      const stream = await getAudioStream();

      setPermission('granted');
      streamRef.current = stream;

      checkForAudioTracks(stream);

      return 'granted' as const;
    } catch (e) {
      setPermission('denied');
      return 'denied' as const;
    }
  }, []);

  return {
    streamRef,
    getStream,
    permission,
  };
};

export { useEncoding };
