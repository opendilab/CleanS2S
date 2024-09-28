// cspell:ignore dataavailable
import Meyda from 'meyda';
import type { MeydaFeaturesObject } from 'meyda';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { MutableRefObject } from 'react';

import { generateEmptyFft } from './generateEmptyFft';

export enum MimeType {
  WEBM = 'audio/webm',
  MP4 = 'audio/mp4',
  WAV = 'audio/wav',
}

export type MicrophoneProps = {
  streamRef: MutableRefObject<MediaStream | null>;
  onAudioCaptured: (b: ArrayBuffer) => void;
  onStartRecording?: () => void;
  onStopRecording?: () => void;
  onError: (message: string) => void;
};

/**
 * Represents a successful result where a compatible MIME type was found.
 * @property {true} success - Indicates a successful result.
 * @property {MimeType} mimeType - The MIME type supported by the browser.
 */
type MimeTypeSuccessResult = { success: true; mimeType: MimeType };

/**
 * Represents a failure result when no compatible MIME type is supported or an error occurs.
 * @property {false} success - Indicates a failure result.
 * @property {Error} error - The error explaining why a compatible MIME type was not found.
 */
type MimeTypeFailureResult = { success: false; error: Error };

/**
 * Union type representing the possible outcomes of checking for a supported MIME type.
 * Could either be a successful or failure result.
 */
type MimeTypeResult = MimeTypeSuccessResult | MimeTypeFailureResult;

/**
 * Checks whether the `MediaRecorder` API is supported in the current environment.
 *
 * @returns {boolean} Returns `true` if the `MediaRecorder` API is supported, otherwise `false`.
 */
function isMediaRecorderSupported(): boolean {
  return typeof MediaRecorder !== 'undefined';
}

/**
 * Finds and returns the first MIME type from the given array that is supported by the `MediaRecorder`.
 *
 * @param {MimeType[]} mimeTypes - An array of MIME types to check for compatibility.
 * @returns {MimeType | null} The first supported MIME type or `null` if none are supported.
 */
function getSupportedMimeType(mimeTypes: MimeType[]): MimeType | null {
  return mimeTypes.find((type) => MediaRecorder.isTypeSupported(type)) || null;
}

/**
 * Determines if the current browser supports any of the predefined audio MIME types
 * (WEBM, MP4, or WAV) via the `MediaRecorder` API.
 *
 * @returns {MimeTypeResult} An object containing the success status and either a supported MIME type or an error.
 * @throws {Error} If the `MediaRecorder` API is not supported by the browser or no compatible types are found.
 */
export function getBrowserSupportedMimeType(): MimeTypeResult {
  // Check if the MediaRecorder API is supported in the current environment.
  if (!isMediaRecorderSupported()) {
    return {
      success: false,
      error: new Error('MediaRecorder is not supported'),
    };
  }

  const COMPATIBLE_MIME_TYPES = [MimeType.WEBM, MimeType.MP4, MimeType.WAV];

  // Find the first compatible MIME type that the browser's MediaRecorder supports.
  const supportedMimeType = getSupportedMimeType(COMPATIBLE_MIME_TYPES);

  // If no compatible MIME type is found, return a failure result with an appropriate error message.
  if (!supportedMimeType) {
    return {
      success: false,
      error: new Error('Browser does not support any compatible mime types'),
    };
  }

  // If a compatible MIME type is found, return a success result with the supported MIME type.
  return {
    success: true,
    mimeType: supportedMimeType,
  };
}

export const useMicrophone = (props: MicrophoneProps) => {
  const { streamRef, onAudioCaptured, onError } = props;
  const [isMuted, setIsMuted] = useState(false);
  const isMutedRef = useRef(isMuted);

  const [fft, setFft] = useState<number[]>(generateEmptyFft());
  // @ts-ignore
  const currentAnalyzer = useRef<Meyda.MeydaAnalyzer | null>(null);
  const mimeTypeRef = useRef<MimeType | null>(null);

  const audioContext = useRef<AudioContext | null>(null);

  const recorder = useRef<ScriptProcessorNode | null>(null);

  const sendAudio = useRef(onAudioCaptured);
  sendAudio.current = onAudioCaptured;

  const dataHandler = useCallback(async (event: BlobEvent) => {
    const blob = event.data;
    try {
      const buffer = await blob.arrayBuffer();
      if (buffer.byteLength > 0) {
        const tmpContext = new AudioContext({ sampleRate: 16000 });
        const audioBuffer = await tmpContext.decodeAudioData(buffer);
        const float32Data = audioBuffer.getChannelData(0);

        sendAudio.current?.(float32Data);
      }
    } catch (err) {
      console.error(err);
    }
  }, []);

  const start = useCallback(() => {
    const stream = streamRef.current;
    if (!stream) {
      throw new Error('No stream connected');
    }

    const context = new AudioContext({ sampleRate: 16000 });
    audioContext.current = context;
    const input = context.createMediaStreamSource(stream);
    const realRecorder = context.createScriptProcessor(1024, 1, 1);
    recorder.current = realRecorder;

    try {
      currentAnalyzer.current = Meyda.createMeydaAnalyzer({
        audioContext: context,
        source: input,
        featureExtractors: ['loudness'],
        callback: (features: MeydaFeaturesObject) => {
          const newFft = features.loudness.specific || [];
          setFft(() => Array.from(newFft));
        },
      });

      currentAnalyzer.current.start();
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : 'Unknown error';
      console.error(`Failed to start mic analyzer: ${message}`);
    }

    recorder.current.onaudioprocess = event => {
      sendAudio.current?.(event.inputBuffer.getChannelData(0));
    };

    input.connect(recorder.current);
    recorder.current.connect(context.destination);


  }, [dataHandler, streamRef, mimeTypeRef]);

  const stop = useCallback(() => {
    try {
      if (currentAnalyzer.current) {
        currentAnalyzer.current.stop();
        currentAnalyzer.current = null;
      }

      if (audioContext.current) {
        void audioContext.current
          .close()
          .then(() => {
            audioContext.current = null;
          })
          .catch(() => {
            // .close() rejects if the audio context is already closed.
            // Therefore, we just need to catch the error, but we don't need to
            // do anything with it.
            return null;
          });
      }

      recorder.current?.disconnect();
      // @ts-ignore
      recorder.current?.removeEventListener('dataavailable', dataHandler);
      recorder.current = null;
      streamRef.current?.getTracks().forEach((track) => track.stop());

      setIsMuted(false);
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Unknown error';
      onError(`Error stopping microphone: ${message}`);
      console.log(e);
      void true;
    }
  }, [dataHandler, onError, streamRef]);

  const mute = useCallback(() => {
    if (currentAnalyzer.current) {
      currentAnalyzer.current.stop();
      setFft(generateEmptyFft());
    }

    streamRef.current?.getTracks().forEach((track) => {
      track.enabled = false;
    });

    isMutedRef.current = true;
    setIsMuted(true);
  }, [streamRef]);

  const unmute = useCallback(() => {
    if (currentAnalyzer.current) {
      currentAnalyzer.current.start();
    }

    streamRef.current?.getTracks().forEach((track) => {
      track.enabled = true;
    });

    isMutedRef.current = false;
    setIsMuted(false);
  }, [streamRef]);

  useEffect(() => {
    return () => {
      try {
        recorder.current?.disconnect();
        // @ts-ignore
        recorder.current?.removeEventListener('dataavailable', dataHandler);

        if (currentAnalyzer.current) {
          currentAnalyzer.current.stop();
          currentAnalyzer.current = null;
        }

        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      } catch (e) {
        console.log(e);
        void true;
      }
    };
  }, [dataHandler, streamRef]);

  useEffect(() => {
    const mimeTypeResult = getBrowserSupportedMimeType();
    if (mimeTypeResult.success) {
      mimeTypeRef.current = mimeTypeResult.mimeType;
    } else {
      onError(mimeTypeResult.error.message);
    }
  }, [onError]);

  return {
    start,
    stop,
    mute,
    unmute,
    isMuted,
    fft,
  };
};
