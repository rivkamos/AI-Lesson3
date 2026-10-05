import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { Header } from '../header/header';
import { Sidebar } from '../sidebar/sidebar';

@Component({
  selector: 'app-shell',
  imports: [RouterOutlet,
    Sidebar,
    Header],
  templateUrl: './shell.html',
  styleUrl: './shell.css',
})
export class Shell {}
