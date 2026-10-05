import { Component, inject } from '@angular/core';
import { DocumentsService } from '../../services/document.service';

@Component({
  selector: 'app-document-upload',
  imports: [],
  templateUrl: './document-upload.html',
  styleUrl: './document-upload.css',
})
export class DocumentUpload {
  private documentsService =
    inject(DocumentsService);

  upload(event: Event) {

    const input =
      event.target as HTMLInputElement;

    const file = input.files?.[0];

    if (!file) return;

    this.documentsService
      .upload(file)
      .subscribe();
  }}
