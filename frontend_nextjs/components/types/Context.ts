export type ContextType = 'editable' | 'persistent' | 'temporary';

export const ContextType = {
  Editable: 'editable',
  Persistent: 'persistent',
  Temporary: 'temporary',
} as const;


export interface Context {
  /**
   * The persistence level of the injected context. Specifies how long the injected context will remain active in the session.
   *
   * There are three possible context types:
   *
   * - **Persistent**: The context is appended to all user messages for the duration of the session.
   *
   * - **Temporary**: The context is appended only to the next user message.
   *
   * - **Editable**: The original context is updated to reflect the new context.
   */
  type?: ContextType;
  /**
   * The context to be injected into the conversation. Helps inform the LLM's response by providing relevant information about the ongoing conversation.
   *
   * This text will be appended to the end of user messages based on the chosen persistence level. For example, if you want to remind EVI of its role as a helpful weather assistant, the context you insert will be appended to the end of user messages as `{Context: You are a helpful weather assistant}`.
   */
  text: string;
}
