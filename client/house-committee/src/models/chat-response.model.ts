export interface ChatSource {
  source: string;
}
export type ChatRole = 'user' | 'assistant';

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
}