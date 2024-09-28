import { AssistantMessage } from './AssistantMessage';
import { ChatMetadata } from './ChatMetadata';


export type JsonMessage = 
    | AssistantMessage
    | ChatMetadata
