import { Context } from './Context';
import { AudioConfiguration } from './AudioConfiguration';

export interface SessionSettings {
  /**
   * The type of message sent through the socket; must be `session_settings` for our server to correctly identify and process it as a Session Settings message.
   *
   * Session settings are temporary and apply only to the current Chat session. These settings can be adjusted dynamically based on the requirements of each session to ensure optimal performance and user experience.
   *
   * For more information, please refer to the [Session Settings section](/docs/empathic-voice-interface-evi/configuration#session-settings) on the EVI Configuration page.
   */
  type: 'session_settings';
  /**
   * Unique identifier for the session. Used to manage conversational state, correlate frontend and backend data, and persist conversations across EVI sessions.
   *
   * If included, the response sent to your backend will include this ID. This allows you to correlate frontend users with their incoming messages.
   *
   * It is recommended to pass a `custom_session_id` if you are using a Custom Language Model. Please see our guide to [using a custom language model](/docs/empathic-voice-interface-evi/custom-language-model) with EVI to learn more.
   */
  customSessionId?: string;
  /**
   * Instructions used to shape EVI’s behavior, responses, and style for the session.
   *
   * When included in a Session Settings message, the provided Prompt overrides the existing one specified in the EVI configuration. If no Prompt was defined in the configuration, this Prompt will be the one used for the session.
   *
   * You can use the Prompt to define a specific goal or role for EVI, specifying how it should act or what it should focus on during the conversation. For example, EVI can be instructed to act as a customer support representative, a fitness coach, or a travel advisor, each with its own set of behaviors and response styles.
   *
   * For help writing a system prompt, see our [Prompting Guide](/docs/empathic-voice-interface-evi/prompting).
   */
  systemPrompt?: string;
  /**
   * Allows developers to inject additional context into the conversation, which is appended to the end of user messages for the session.
   *
   * When included in a Session Settings message, the provided context can be used to remind the LLM of its role in every user message, prevent it from forgetting important details, or add new relevant information to the conversation.
   *
   * Set to `null` to disable context injection.
   */
  context?: Context;
  /**
   * Configuration details for the audio input used during the session. Ensures the audio is being correctly set up for processing.
   *
   * This optional field is only required when the audio input is encoded in PCM Linear 16 (16-bit, little-endian, signed PCM WAV data). For detailed instructions on how to configure session settings for PCM Linear 16 audio, please refer to the [Session Settings section](/docs/empathic-voice-interface-evi/configuration#session-settings) on the EVI Configuration page.
   */
  audio?: AudioConfiguration;
  /**
   * Third party API key for the supplemental language model.
   *
   * When provided, EVI will use this key for the supplemental LLM. This allows you to bypass rate limits and utilize your own API key as needed.
   */
  languageModelApiKey?: string;
  metadata?: Record<string, unknown>;
  /** Dynamic values that can be used to populate EVI prompts. */
  variables?: Record<string, string>;
}
