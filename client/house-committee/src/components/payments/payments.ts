import { Component } from '@angular/core';

@Component({
  selector: 'app-payments',
  imports: [],
  templateUrl: './payments.html',
  styleUrl: './payments.css',
})
export class Payments {
   payments = [
    { name: 'משה כהן', amount: 250, status: 'שולם' },
    { name: 'דנה לוי', amount: 250, status: 'בהמתנה' }
  ];
}
