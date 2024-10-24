import { useCallback, useRef, useState } from 'react';

import { AudioOutput } from './types';
import { convertLinearFrequenciesToBark } from './convertFrequencyScale';
import { generateEmptyFft } from './generateEmptyFft';


export function convertBase64ToBlob(base64: string, contentType: string): Blob {
  // Decode base64 string to a binary string
  const binaryString = window.atob(base64);

  // Create a Uint8Array with the same length as the binary string
  const byteArray = new Uint8Array(binaryString.length);

  // Fill the Uint8Array by converting each character's Unicode value to a byte
  for (let i = 0; i < binaryString.length; i++) {
    byteArray[i] = binaryString.charCodeAt(i);
  }

  // Create and return a Blob with the specified content type
  return new Blob([byteArray], { type: contentType });
}


function downloadWav(audioBuffer: AudioBuffer, fileName: string) {
  const wavData = audioBufferToWav(audioBuffer);

  const blob = new Blob([wavData], { type: 'audio/wav' });

  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();

  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function audioBufferToWav(audioBuffer: AudioBuffer) {
  const numChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const bufferLength = audioBuffer.length;

  const arrayBuffer = new ArrayBuffer(44 + bufferLength * numChannels * 2);
  const view = new DataView(arrayBuffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 32 + bufferLength * numChannels * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * 2, true);
  view.setUint16(32, numChannels * 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, bufferLength * numChannels * 2, true);

  const channelData = new Float32Array(bufferLength);
  for (let channel = 0; channel < numChannels; channel++) {
    const channelBuffer = audioBuffer.getChannelData(channel);
    for (let i = 0; i < bufferLength; i++) {
      channelData[i] = channelBuffer[i];
    }
    for (let i = 0; i < bufferLength; i++) {
      const sample = Math.max(-1, Math.min(1, channelData[i]));
      view.setInt16(44 + (i * numChannels + channel) * 2, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
    }
  }

  return arrayBuffer;
}

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

export const useSoundPlayer = (props: {
  onError: (message: string) => void;
  onPlayAudio: (id: string) => void;
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isAudioMuted, setIsAudioMuted] = useState(false);
  const [playedID, setPlayedID] = useState("placeholder")
  const [fft, setFft] = useState<number[]>(generateEmptyFft());

  const stopPlaybackRef = useRef<() => void>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const analyserNode = useRef<AnalyserNode | null>(null);
  const gainNode = useRef<GainNode | null>(null);
  const isInitialized = useRef(false);

  const clipQueue = useRef<
    Array<{
      id: string;
      buffer: AudioBuffer;
    }>
  >([]);
  const historyClipQueue = useRef<
    Array<{
      id: string;
      buffer: AudioBuffer;
    }>
  >([]);
  const isProcessing = useRef(false);
  const currentlyPlayingAudioBuffer = useRef<AudioBufferSourceNode | null>(
    null,
  );
  const frequencyDataIntervalId = useRef<number | null>(null);

  const onPlayAudio = useRef<typeof props.onPlayAudio>(props.onPlayAudio);
  onPlayAudio.current = props.onPlayAudio;

  const onError = useRef<typeof props.onError>(props.onError);
  onError.current = props.onError;

  const playNextClip = useCallback((targetClip?: {id: string, buffer: AudioBuffer}) => {
    if (analyserNode.current === null || audioContext.current === null) {
      onError.current('Audio environment is not initialized');
      return;
    }

    if (!targetClip && (clipQueue.current.length === 0 || isProcessing.current)) {
      return;
    }

    var nextClip = null;
    if (targetClip) {
        nextClip = targetClip;
        // stop the current audio buffer if it's playing
        if (stopPlaybackRef.current) {
          stopPlaybackRef.current();
        }
    } else {
        nextClip = clipQueue.current.shift();
    }
    if (!nextClip) return;

    setPlayedID(nextClip.id);
    isProcessing.current = true;
    setIsPlaying(true);

    // Use AudioBufferSourceNode for audio playback.
    // Safari suffered a truncation issue using HTML5 audio playback
    const bufferSource = audioContext.current.createBufferSource();

    bufferSource.buffer = nextClip.buffer;

    bufferSource.connect(analyserNode.current);
    // @ts-ignore
    stopPlaybackRef.current = () => {bufferSource.stop()}

    currentlyPlayingAudioBuffer.current = bufferSource;

    const updateFrequencyData = () => {
      try {
        const bufferSampleRate = bufferSource.buffer?.sampleRate;

        if (!analyserNode.current || typeof bufferSampleRate === 'undefined')
          return;

        const dataArray = new Uint8Array(
          analyserNode.current.frequencyBinCount,
        ); // frequencyBinCount is 1/2 of fftSize
        analyserNode.current.getByteFrequencyData(dataArray); // Using getByteFrequencyData for performance

        const barkFrequencies = convertLinearFrequenciesToBark(
          dataArray,
          bufferSampleRate,
        );
        setFft(() => barkFrequencies);
      } catch (e) {
        setFft(generateEmptyFft());
      }
    };

    frequencyDataIntervalId.current = window.setInterval(
      updateFrequencyData,
      5,
    );

    bufferSource.start(0);
    onPlayAudio.current(nextClip.id);

    bufferSource.onended = () => {
      if (frequencyDataIntervalId.current) {
        clearInterval(frequencyDataIntervalId.current);
      }
      setFft(generateEmptyFft());
      bufferSource.disconnect();
      isProcessing.current = false;
      setIsPlaying(false);
      currentlyPlayingAudioBuffer.current = null;
      playNextClip();
    };
  }, []);

  const initPlayer = useCallback(() => {
    const initAudioContext = new AudioContext();
    audioContext.current = initAudioContext;

    // Use AnalyserNode to get fft frequency data for visualizations
    const analyser = initAudioContext.createAnalyser();
    // Use GainNode to adjust volume
    const gain = initAudioContext.createGain();

    analyser.fftSize = 2048; // Must be a power of 2
    analyser.connect(gain);
    gain.connect(initAudioContext.destination);

    analyserNode.current = analyser;
    gainNode.current = gain;

    isInitialized.current = true;
  }, []);

  const addToQueue = useCallback(
    async (message: AudioOutput) => {
      if (!isInitialized.current || !audioContext.current) {
        onError.current('Audio player has not been initialized');
        return;
      }

      try {
        const arrayBuffer = message.data;
        const numberOfChannels = 1;
        const sampleRate = 16000;
        const numberOfFrames = arrayBuffer.byteLength / (numberOfChannels * 2);

        const audioBuffer = audioContext.current.createBuffer(numberOfChannels, numberOfFrames, sampleRate);

        for (let channel = 0; channel < numberOfChannels; channel++) {
            const nowBuffering = audioBuffer.getChannelData(channel);
            const int16Array = new Int16Array(arrayBuffer, channel * numberOfFrames * 2, numberOfFrames);
            const float32Array = new Float32Array(int16Array.length);

            for (let i = 0; i < int16Array.length; i++) {
              float32Array[i] = int16Array[i] / 32768.0;
            }

            nowBuffering.set(float32Array);
        }

        clipQueue.current.push({
          id: message.id,
          buffer: audioBuffer,
        });
        historyClipQueue.current.push({
          id: message.id,
          buffer: audioBuffer,
        });

        // playNextClip will iterate the clipQueue upon finishing the playback of the current audio clip, so we can
        // just call playNextClip here if it's the only one in the queue
        if (clipQueue.current.length === 1) {
          playNextClip();
        }
      } catch (e) {
        const eMessage = e instanceof Error ? e.message : 'Unknown error';
        onError.current(`Failed to add clip to queue: ${eMessage}`);
      }
    },
    [playNextClip],
  );

  const stopAll = useCallback(() => {
    isInitialized.current = false;
    isProcessing.current = false;
    setIsPlaying(false);

    if (frequencyDataIntervalId.current) {
      window.clearInterval(frequencyDataIntervalId.current);
    }

    if (currentlyPlayingAudioBuffer.current) {
      currentlyPlayingAudioBuffer.current.disconnect();
      currentlyPlayingAudioBuffer.current = null;
    }

    if (analyserNode.current) {
      analyserNode.current.disconnect();
      analyserNode.current = null;
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

    clipQueue.current = [];
    historyClipQueue.current = [];
    setFft(generateEmptyFft());
  }, []);

  const clearQueue = useCallback(() => {
    if (currentlyPlayingAudioBuffer.current) {
      currentlyPlayingAudioBuffer.current.stop();
      currentlyPlayingAudioBuffer.current = null;
    }

    // don't clear historyClipQueue in this callback
    clipQueue.current = [];
    isProcessing.current = false;
    setIsPlaying(false);
    setFft(generateEmptyFft());
  }, []);

  const muteAudio = useCallback(() => {
    if (gainNode.current && audioContext.current) {
      gainNode.current.gain.setValueAtTime(0, audioContext.current.currentTime);
      setIsAudioMuted(true);
    }
  }, []);

  const unmuteAudio = useCallback(() => {
    if (gainNode.current && audioContext.current) {
      gainNode.current.gain.setValueAtTime(1, audioContext.current.currentTime);
      setIsAudioMuted(false);
    }
  }, []);

  const downloadAudio = useCallback((aid: string) => {
    const foundClip = historyClipQueue.current.find(clip => clip.id === aid);

    if (foundClip) {
      downloadWav(foundClip.buffer, `audio_${aid}_${Date.now()}.wav`);
    } else {
      console.log("Clip not found", aid);
    }
  }, [])

  const replayAudio = useCallback((aid: string) => {
    if (clipQueue.current.length !== 0) {
      console.log("Can't replay audio when agent is playing")
      return;
    }
    const foundClip = historyClipQueue.current.find(clip => clip.id === aid);

    if (foundClip) {
      playNextClip(foundClip);
    } else {
      console.log("Clip not found", aid);
    }
  }, [])

  return {
    addToQueue,
    fft,
    initPlayer,
    isPlaying,
    isAudioMuted,
    muteAudio,
    unmuteAudio,
    stopAll,
    clearQueue,
    downloadAudio,
    replayAudio,
    playedID,
  };
};
