import { JsonMessage, AssistantMessage, UserMessage, AudioOutput, SessionSettings, ChatMetadata, UserInterruption } from './types';
import React, {
  createContext,
  FC,
  PropsWithChildren,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

import { ConnectionMessage } from './connection-message';
import { noop } from './noop';
import { useCallDuration } from './useCallDuration';
import { useEncoding } from './useEncoding';
import { useMessages } from './useMessages';
import { useMicrophone } from './useMicrophone';
import { useSoundPlayer } from './useSoundPlayer';
import { ChatSocket } from './ChatSocket';
import {
  SocketConfig,
  useVoiceClient,
  type VoiceReadyState,
} from './useVoiceClient';

type VoiceError =
  | { type: 'socket_error'; message: string; error?: Error }
  | { type: 'audio_error'; message: string; error?: Error }
  | { type: 'mic_error'; message: string; error?: Error };

type VoiceStatus =
  | {
      value: 'disconnected' | 'connecting' | 'connected';
      reason?: never;
    }
  | {
      value: 'error';
      reason: string;
    };

export type VoiceContextType = {
  connect: () => Promise<void>;
  disconnect: () => void;
  fft: number[];
  isMuted: boolean;
  isAudioMuted: boolean;
  isPlaying: boolean;
  messages: (
    | JsonMessage
    | ConnectionMessage
    | UserMessage
    | AssistantMessage
    | UserInterruption
  )[];
  lastVoiceMessage: AssistantMessage | null;
  lastUserMessage: UserMessage | null;
  clearMessages: () => void;
  mute: () => void;
  unmute: () => void;
  muteAudio: () => void;
  unmuteAudio: () => void;
  downloadAudio: (aid: string) => void;
  replayAudio: (aid: string) => void;
  playedID: string;
  readyState: VoiceReadyState;
  sendUserInput: (text: string) => void;
  sendAssistantInput: (text: string) => void;
  sendSessionSettings: ChatSocket['sendSessionSettings'];
  sendPauseAssistantMessage: ChatSocket['pauseAssistant'];
  sendResumeAssistantMessage: ChatSocket['resumeAssistant'];
  status: VoiceStatus;
  micFft: number[];
  error: VoiceError | null;
  isAudioError: boolean;
  isError: boolean;
  isMicrophoneError: boolean;
  isSocketError: boolean;
  callDurationTimestamp: string | null;
  chatMetadata: ChatMetadata | null;
  clearCurrentTopic: () => void;
};

const VoiceContext = createContext<VoiceContextType | null>(null);

export type VoiceProviderProps = PropsWithChildren<SocketConfig> & {
  sessionSettings?: SessionSettings;
  onMessage?: (message: any) => void;
  onError?: (err: VoiceError) => void;
  onOpen?: () => void;
  onClose?: () => void;
  /**
   * @default true
   * @description Clear messages when the voice is disconnected.
   */
  clearMessagesOnDisconnect?: boolean;
  /**
   * @default 100
   * @description The maximum number of messages to keep in memory.
   */
  messageHistoryLimit?: number;
};

export const useVoice = () => {
  const ctx = useContext(VoiceContext);
  if (!ctx) {
    throw new Error('useVoice must be used within an VoiceProvider');
  }
  return ctx;
};

export const VoiceProvider: FC<VoiceProviderProps> = ({
  children,
  clearMessagesOnDisconnect = true,
  messageHistoryLimit = 100,
  sessionSettings,
  ...props
}) => {
  const {
    timestamp: callDurationTimestamp,
    start: startTimer,
    stop: stopTimer,
  } = useCallDuration();

  const [status, setStatus] = useState<VoiceStatus>({
    value: 'disconnected',
  });

  // error handling
  const [error, setError] = useState<VoiceError | null>(null);
  const isError = error !== null;
  const isMicrophoneError = error?.type === 'mic_error';
  const isSocketError = error?.type === 'socket_error';
  const isAudioError = error?.type === 'audio_error';

  const onError = useRef(props.onError ?? noop);
  onError.current = props.onError ?? noop;

  const onClose = useRef(props.onClose ?? noop);
  onClose.current = props.onClose ?? noop;

  const messageStore = useMessages({
    sendMessageToParent: props.onMessage,
    messageHistoryLimit,
  });

  const updateError = useCallback((err: VoiceError | null) => {
    setError(err);
    if (err !== null) {
      onError.current?.(err);
    }
  }, []);

  const onClientError: NonNullable<
    Parameters<typeof useVoiceClient>[0]['onError']
  > = useCallback(
    (message, err) => {
      stopTimer();
      updateError({ type: 'socket_error', message, error: err });
    },
    [stopTimer, updateError],
  );

  const config = props;

  const player = useSoundPlayer({
    onError: (message) => {
      updateError({ type: 'audio_error', message });
    },
    onPlayAudio: (id: string) => {
      messageStore.onPlayAudio(id);
    },
  });

  const { streamRef, getStream, permission: micPermission } = useEncoding();

  const client = useVoiceClient({
    onAudioMessage: (message: AudioOutput) => {
      player.addToQueue(message);
    },
    onMessage: useCallback(
      (message: JsonMessage & { receivedAt: Date } | AssistantMessage | UserMessage) => {
        // store message
        messageStore.onMessage(message);

        if (
          message.type === 'user_message'
        ) {
          player.clearQueue();
        }

      },
      [messageStore, player],
    ),
    onVAD: () => {
      if (player.isPlaying) {
        player.clearQueue();
      }
    },
    onError: onClientError,
    onOpen: useCallback(() => {
      startTimer();
      messageStore.createConnectMessage();
      props.onOpen?.();
    }, [messageStore, props, startTimer]),
    onClose: useCallback<
      NonNullable<any>
    >(
      (event: Event) => {
        stopTimer();
        messageStore.createDisconnectMessage();
        onClose.current?.();
      },
      [messageStore, stopTimer],
    ),
  });

  const mic = useMicrophone({
    streamRef,
    onAudioCaptured: useCallback((arrayBuffer) => {
      try {
        client.sendAudio(arrayBuffer, player.isPlaying);
      } catch (e) {
        const message = e instanceof Error ? e.message : 'Unknown error';
        updateError({ type: 'socket_error', message });
      }
    }, [player.isPlaying]),
    onError: useCallback(
      (message) => {
        updateError({ type: 'mic_error', message });
      },
      [updateError],
    ),
  });

  const connect = useCallback(async () => {
    updateError(null);
    setStatus({ value: 'connecting' });
    const permission = await getStream();

    if (permission === 'denied') {
      const error: VoiceError = {
        type: 'mic_error',
        message: 'Microphone permission denied',
      };
      updateError(error);
      return Promise.reject(error);
    }

    try {
      await client
        .connect({
          ...config,
        })
        .then(() => {
          if (
            sessionSettings !== undefined &&
            Object.keys(sessionSettings).length > 0
          ) {
            client.sendSessionSettings(sessionSettings);
          }
        });
    } catch (e) {
      const error: VoiceError = {
        type: 'socket_error',
        message: 'We could not connect to the voice. Please try again.',
      };
      updateError(error);
      return Promise.reject(error);
    }

    try {
      const [micPromise, playerPromise] = await Promise.allSettled([
        mic.start(),
        player.initPlayer(),
      ]);

      if (
        micPromise.status === 'fulfilled' &&
        playerPromise.status === 'fulfilled'
      ) {
        setStatus({ value: 'connected' });
      }
    } catch (e) {
      const error: VoiceError = {
        type: 'audio_error',
        message:
          e instanceof Error
            ? e.message
            : 'We could not connect to audio. Please try again.',
      };
      updateError(error);
    }
  }, [client, config, getStream, mic, player, sessionSettings, updateError]);

  const disconnectFromVoice = useCallback(() => {
    client.disconnect();
    player.stopAll();
    mic.stop();
    if (clearMessagesOnDisconnect) {
      messageStore.clearMessages();
    }
  }, [
    client,
    player,
    mic,
    clearMessagesOnDisconnect,
    messageStore,
  ]);

  const clearCurrentTopic = useCallback(() => {
    player.stopAll();
    if (clearMessagesOnDisconnect) {
      messageStore.clearMessages();
    }
  }, []);

  const disconnect = useCallback(
    (disconnectOnError?: boolean) => {
      if (micPermission === 'denied') {
        setStatus({ value: 'error', reason: 'Microphone permission denied' });
      }

      stopTimer();

      disconnectFromVoice();

      if (status.value !== 'error' && !disconnectOnError) {
        // if status was 'error', keep the error status so we can show the error message to the end user.
        // otherwise, set status to 'disconnected'
        setStatus({ value: 'disconnected' });
      }
    },
    [micPermission, stopTimer, disconnectFromVoice, status.value],
  );

  useEffect(() => {
    if (
      error !== null &&
      status.value !== 'error' &&
      status.value !== 'disconnected'
    ) {
      // If the status is ever set to `error`, disconnect the voice.
      setStatus({ value: 'error', reason: error.message });
      disconnectFromVoice();
    }
  }, [status.value, disconnect, disconnectFromVoice, error]);

  useEffect(() => {
    // disconnect from socket when the voice provider component unmounts
    return () => {
      disconnectFromVoice();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const ctx = useMemo(
    () =>
      ({
        connect,
        disconnect,
        fft: player.fft,
        micFft: mic.fft,
        isMuted: mic.isMuted,
        isAudioMuted: player.isAudioMuted,
        isPlaying: player.isPlaying,
        messages: messageStore.messages,
        lastVoiceMessage: messageStore.lastVoiceMessage,
        lastUserMessage: messageStore.lastUserMessage,
        clearMessages: messageStore.clearMessages,
        mute: mic.mute,
        muteAudio: player.muteAudio,
        downloadAudio: player.downloadAudio,
        replayAudio: player.replayAudio,
        playedID: player.playedID,
        readyState: client.readyState,
        sendUserInput: client.sendUserInput,
        sendAssistantInput: client.sendAssistantInput,
        sendSessionSettings: client.sendSessionSettings,
        sendPauseAssistantMessage: client.sendPauseAssistantMessage,
        sendResumeAssistantMessage: client.sendResumeAssistantMessage,
        status,
        unmute: mic.unmute,
        unmuteAudio: player.unmuteAudio,
        error,
        isAudioError,
        isError,
        isMicrophoneError,
        isSocketError,
        callDurationTimestamp,
        chatMetadata: messageStore.chatMetadata,
        clearCurrentTopic: clearCurrentTopic,
      }) satisfies VoiceContextType,
    [
      connect,
      disconnect,
      player.fft,
      player.isPlaying,
      player.isAudioMuted,
      player.muteAudio,
      player.unmuteAudio,
      player.downloadAudio,
      player.replayAudio,
      player.playedID,
      mic.fft,
      mic.isMuted,
      mic.mute,
      mic.unmute,
      messageStore.messages,
      messageStore.lastVoiceMessage,
      messageStore.lastUserMessage,
      messageStore.clearMessages,
      client.readyState,
      client.sendUserInput,
      client.sendAssistantInput,
      client.sendSessionSettings,
      client.sendPauseAssistantMessage,
      client.sendResumeAssistantMessage,
      status,
      error,
      isAudioError,
      isError,
      isMicrophoneError,
      isSocketError,
      callDurationTimestamp,
      messageStore.chatMetadata,
      clearCurrentTopic,
    ],
  );

  return <VoiceContext.Provider value={ctx}>{children}</VoiceContext.Provider>;
};
