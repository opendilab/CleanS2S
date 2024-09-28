export interface ChatMetadata {
  /**
   * The type of message sent through the socket; for a Chat Metadata message, this must be `chat_metadata`.
   *
   * The Chat Metadata message is the first message you receive after establishing a connection with EVI and contains important identifiers for the current Chat session.
   */
  type: 'chat_metadata';
  /** Used to manage conversational state, correlate frontend and backend data, and persist conversations across EVI sessions. */
  customSessionId?: string;
  /**
   * ID of the Chat Group.
   *
   * Used to resume a Chat when passed in the [resumed_chat_group_id](/reference/empathic-voice-interface-evi/chat/chat#request.query.resumed_chat_group_id) query parameter of a subsequent connection request. This allows EVI to continue the conversation from where it left off within the Chat Group.
   *
   * Learn more about [supporting chat resumability](/docs/empathic-voice-interface-evi/faq#does-evi-support-chat-resumability) from the EVI FAQ.
   */
  chatGroupId: string;
  /** ID of the Chat session. Allows the Chat session to be tracked and referenced. */
  chatId: string;
  /** ID of the initiating request. */
  requestId?: string;
}
