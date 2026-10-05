import { Component } from '@angular/core';

@Component({
  selector: 'app-buildings',
  imports: [],
  templateUrl: './buildings.html',
  styleUrl: './buildings.css',
})
export class Buildings {
   buildings = [
    { name: 'בניין A', address: 'רחוב הרצל 10' },
    { name: 'בניין B', address: 'ויצמן 22' }
  ];

  addBuilding() {
    console.log('add building');
  }
}
