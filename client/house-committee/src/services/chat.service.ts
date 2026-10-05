import { HttpClient } from "@angular/common/http";
import { inject, Injectable } from "@angular/core";
import { Observable } from 'rxjs';
import { ChatResponse } from "../models/chat-response.model";


@Injectable({
  providedIn: 'root'
})
export class ChatService {

  private readonly http = inject(HttpClient);

  ask(question: string): Observable<ChatResponse> {

    return this.http.post<ChatResponse>(
      'http://localhost:8000/api/chat',
      {
        question
      }
    );
  }
}