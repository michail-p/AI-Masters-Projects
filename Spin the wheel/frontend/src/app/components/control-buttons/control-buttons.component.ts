import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';
import { Observable } from 'rxjs';
import { LoadingService } from '../../services/loading.service';

@Component({
  selector: 'app-control-buttons',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './control-buttons.component.html',
  styleUrls: ['./control-buttons.component.css']
})
export class ControlButtonsComponent {
  @Input() canCompare: boolean = false;
  @Input() canRestart: boolean = false;
  @Input() isComparingScenarios: boolean = false;
  @Output() compareClicked = new EventEmitter<void>();
  @Output() restartClicked = new EventEmitter<void>();
  @Output() compareScenarioClicked = new EventEmitter<void>();

  readonly loading$: Observable<boolean>;

  constructor(private loadingService: LoadingService) {
    this.loading$ = loadingService.loading$;
  }

  compare() {
    this.compareClicked.emit();
  }

  restart() {
    this.restartClicked.emit();
  }

  compareScenario(){
    this.compareScenarioClicked.emit();
  }

}
