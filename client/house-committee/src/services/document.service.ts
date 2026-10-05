import { HttpClient } from "@angular/common/http";
import { Injectable } from "@angular/core";
import { inject } from "@angular/core/primitives/di";

@Injectable({
  providedIn: 'root'
})
export class DocumentsService {

  private readonly http = inject(HttpClient);

  upload(file: File) {

    const formData = new FormData();

    formData.append('file', file);

    return this.http.post(
      'http://localhost:8000/api/documents/upload',
      formData
    );
  }
}