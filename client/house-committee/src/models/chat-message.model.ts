
import { ChatRole, ChatSource } from "./chat-response.model";

export interface ChatMessage {
  role: ChatRole;
  content: string;
  sources?: ChatSource[];
}