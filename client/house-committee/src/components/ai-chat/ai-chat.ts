import { Component, inject, signal } from '@angular/core';
import { ChatService } from '../../services/chat.service';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import {  ElementRef,  ViewChild} from '@angular/core';
import { ChatMessage } from '../../models/chat-message.model';
import { ChatResponse } from '../../models/chat-response.model';

@Component({
  selector: 'app-ai-chat',
  imports: [CommonModule,
    FormsModule,
  MatIconModule,
    MatProgressSpinnerModule],
  templateUrl: './ai-chat.html',
  styleUrl: './ai-chat.css',
})

export class AiChat {

private chatService = inject(ChatService);

  @ViewChild('messagesContainer')
  messagesContainer!: ElementRef<HTMLDivElement>;
messageView: ChatResponse | null = null;
  question = signal('');

  loading = signal(false);

  messages = signal<ChatMessage[]>([
    {
      role: 'assistant',
      content:
        'שלום 👋 אני עוזר ועד הבית. שאלי אותי על תקנונים, פרוטוקולים, תשלומים ומסמכים.'
    }
  ]);

  send(): void {

    const question = this.question().trim();

    if (!question) {
      return;
    }

    this.messages.update(messages => [
      ...messages,
      {
        role: 'user',
        content: question
      }
    ]);

    this.question.set('');

    this.scrollToBottom();

    this.loading.set(true);

    this.chatService.ask(question).subscribe({
  next: (response: ChatResponse) => {

    const message: ChatMessage = {
      role: 'assistant',
      content: response.answer,
      sources: [...new Set(response.sources.map(s => s.source))]
        .map(source => ({ source }))
    };

    this.messages.update(messages => [
      ...messages,
      message
    ]);

    this.loading.set(false);
    this.scrollToBottom();
  },

  error: (error) => {
    console.error(error);

    this.messages.update(messages => [
      ...messages,
      {
        role: 'assistant',
        content: 'אירעה שגיאה בעת הפנייה לשרת.',
        sources: []
      }
    ]);

    this.loading.set(false);
    this.scrollToBottom();
  }
});
  }

  private scrollToBottom(): void {

    setTimeout(() => {

      if (!this.messagesContainer) {
        return;
      }

      this.messagesContainer.nativeElement.scrollTop =
        this.messagesContainer.nativeElement.scrollHeight;

    }, 50);
  }
}