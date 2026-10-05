import { Component } from '@angular/core';

@Component({
  selector: 'app-tenants',
  imports: [],
  templateUrl: './tenants.html',
  styleUrl: './tenants.css',
})
export class Tenants {
  tenants = [
    { name: 'משה כהן', apartment: '1A', phone: '050-0000000' },
    { name: 'דנה לוי', apartment: '2B', phone: '051-1111111' }
  ];}
