import { ChatMessage } from './ChatMessage';

export interface AssistantMessage {
  /**
   * The type of message sent through the socket; for an Assistant Message, this must be `assistant_message`.
   *
   * This message contains both a transcript of the assistant’s response and the expression measurement predictions of the assistant’s audio output.
   */
  type: 'assistant_message' | 'assistant_notend_message';
  /** Used to manage conversational state, correlate frontend and backend data, and persist conversations across EVI sessions. */
  customSessionId?: string;
  /** ID of the assistant message. Allows the Assistant Message to be tracked and referenced. */
  id?: string;
  /** Transcript of the message. */
  message: ChatMessage;
  /** Indicates if this message was inserted into the conversation as text from an [Assistant Input message](/reference/empathic-voice-interface-evi/chat/chat#send.Assistant%20Input.text). */
  fromText: boolean;
  receivedAt?: Date;
  end?: boolean;
}
