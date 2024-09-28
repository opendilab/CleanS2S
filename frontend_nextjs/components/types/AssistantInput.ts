export interface AssistantInput {
  /** The type of message sent through the socket; must be `assistant_input` for our server to correctly identify and process it as an Assistant Input message. */
  type: 'assistant_input';
  /** Used to manage conversational state, correlate frontend and backend data, and persist conversations across EVI sessions. */
  customSessionId?: string;
  /**
   * Assistant text to synthesize into spoken audio and insert into the conversation.
   *
   * EVI uses this text to generate spoken audio using our proprietary expressive text-to-speech model. Our model adds appropriate emotional inflections and tones to the text based on the user’s expressions and the context of the conversation. The synthesized audio is streamed back to the user as an [Assistant Message](/reference/empathic-voice-interface-evi/chat/chat#receive.Assistant%20Message.type).
   */
  text: string;
}
