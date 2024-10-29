import { AssistantMessage, UserMessage, ChatMetadata, JsonMessage, UserInterruption, PostAssistantMessage } from './types';
import { useCallback, useState } from 'react';

import type { ConnectionMessage } from './connection-message';
import { keepLastN } from '../utils';

export const useMessages = ({
  sendMessageToParent,
  messageHistoryLimit,
}: {
  sendMessageToParent?: (
    message: JsonMessage | AssistantMessage | UserMessage | UserInterruption | PostAssistantMessage,
  ) => void;
  messageHistoryLimit: number;
}) => {
  const [voiceMessageMap, setVoiceMessageMap] = useState<
    Record<string, AssistantMessage & { receivedAt: Date }>
  >({});
  const [postMessage, setPostMessage] = useState<PostAssistantMessage | null>(null);

  const [messages, setMessages] = useState<
    Array<
      | JsonMessage
      | ConnectionMessage
      | AssistantMessage
      | UserMessage
      | UserInterruption
      | PostAssistantMessage
    >
  >([]);

  const [lastVoiceMessage, setLastVoiceMessage] =
    useState<AssistantMessage | null>(null);
  const [lastUserMessage, setLastUserMessage] =
    useState<UserMessage | null>(null);

  const [chatMetadata, setChatMetadata] =
    useState<ChatMetadata | null>(null);

  const createConnectMessage = useCallback(() => {
    setMessages((prev) =>
      prev.concat([
        {
          type: 'socket_connected',
          receivedAt: new Date(),
        },
      ]),
    );
  }, []);

  const createDisconnectMessage = useCallback(() => {
    setMessages((prev) =>
      prev.concat([
        {
          type: 'socket_disconnected',
          receivedAt: new Date(),
        },
      ]),
    );
  }, []);

  const onMessage = useCallback(
    (message: JsonMessage | AssistantMessage | UserMessage | UserInterruption | PostAssistantMessage ) => {
      /* 
      1. message comes in from the backend
        - if the message IS NOT AssistantTranscriptMessage, store in `messages` immediately  
        - if the message is an AssistantTranscriptMessage, stored in `voiceMessageMap`
      2. audio clip plays
        - find the AssistantTranscriptMessage with a matching ID, and store it in `messages`
        - remove the AssistantTranscriptMessage from `voiceMessageMap`
    */
      switch (message.type) {
        case 'assistant_message':
          // for assistant messages, `sendMessageToParent` is called in `onPlayAudio`
          // in order to line up the transcript event with the correct audio clip
          // @ts-ignore
          setVoiceMessageMap((prev) => ({
            ...prev,
            [`${message.id}`]: message,
          }));
          break;
        case 'post_assistant_message':
          setPostMessage(message);
          break;
        case 'user_message':
          sendMessageToParent?.(message);
          setLastUserMessage(message);
          setMessages((prev) => {
            return keepLastN(messageHistoryLimit, prev.concat([message]));
          });
          break;
        case 'user_vad_message':
        case 'assistant_notend_message':
          sendMessageToParent?.(message);
          setMessages((prev) => {
            return keepLastN(messageHistoryLimit, prev.concat([message]));
          });
          break;
        case 'user_interruption':
        //case 'error':
          sendMessageToParent?.(message);
          setMessages((prev) => {
            return keepLastN(messageHistoryLimit, prev.concat([message]));
          });
          break;
        case 'chat_metadata':
          sendMessageToParent?.(message);
          setMessages((prev) => {
            return keepLastN(messageHistoryLimit, prev.concat([message]));
          });
          setChatMetadata(message);
          break;
        default:
          break;
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    },
    [],
  );

  const onPlayAudio = useCallback(
    (id: string) => {
      const matchingTranscript = voiceMessageMap[id];
      if (matchingTranscript) {
        sendMessageToParent?.(matchingTranscript);
        setLastVoiceMessage(matchingTranscript);
        setMessages((prev) => {
          if (matchingTranscript.end) {
            if (postMessage) {
              sendMessageToParent?.(postMessage);
              return keepLastN(
                messageHistoryLimit,
                prev.concat([matchingTranscript, postMessage]),
              );
              setPostMessage(null);
            } else {
              return keepLastN(
                messageHistoryLimit,
                prev.concat([matchingTranscript]),
              );
            }
          } else {
            const notEndedMessage: AssistantMessage = {
              'type': 'assistant_notend_message',
              id: 'notend' + matchingTranscript.id,
              fromText: false,
              message: {
                role: 'assistant',
                content: '',
              },
              receivedAt: new Date(),
              end: false,
            }
            return keepLastN(
              messageHistoryLimit,
              prev.concat([matchingTranscript, notEndedMessage]),
            );
          }
        });
        // remove the message from the map to ensure we don't
        // accidentally push it to the messages array more than once
        setVoiceMessageMap((prev) => {
          const newMap = { ...prev };
          delete newMap[id];
          return newMap;
        });
      }
    },
    [voiceMessageMap, sendMessageToParent, messageHistoryLimit, postMessage],
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    setLastVoiceMessage(null);
    setLastUserMessage(null);
    setVoiceMessageMap({});
  }, []);

  return {
    createConnectMessage,
    createDisconnectMessage,
    onMessage,
    onPlayAudio,
    clearMessages,
    messages,
    lastVoiceMessage,
    lastUserMessage,
    chatMetadata,
  };
};
