import { CommonModule } from '@angular/common';
import { Component, OnInit, ViewChild } from '@angular/core';
import { SlotMachineComponent } from './components/slot-machine/slot-machine.component';
import { ParameterizationService } from './services/parameterization.service';
import { Observable } from 'rxjs';
import { CompareScenariosRequest, Gender } from './api/models';
import { ControlButtonsComponent } from './components/control-buttons/control-buttons.component';
import { SpinService } from './services/spin.service';
import { CompareScenarioComponent } from './components/compare-scenario/compare-scenario.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, SlotMachineComponent, ControlButtonsComponent, CompareScenarioComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent implements OnInit {
  @ViewChild('slot1') slot1!: SlotMachineComponent;
  @ViewChild('slot2') slot2!: SlotMachineComponent;
  readonly selectPlaces$: Observable<string[]>
  readonly selectTimes$: Observable<number[]>
  readonly selectGenders$: Observable<Gender[]>
  compareScenarioBody: CompareScenariosRequest | undefined
  isComparing: boolean = false
  isComparingScenarios: boolean = false
  slot1Visible: boolean = false;
  slot2Visible: boolean = false;
  restartFlag: boolean = false;

  constructor(private parameterizationService: ParameterizationService, private spinService: SpinService) {
    this.selectPlaces$ = parameterizationService.selectPlaces$;
    this.selectTimes$ = parameterizationService.selectTimes$;
    this.selectGenders$ = parameterizationService.selectGenders$;
  }

  ngOnInit() {
    this.parameterizationService.getGenders();
    this.parameterizationService.getPlaces();
    this.parameterizationService.getTimes();
  }

  onCompareClicked() {
    this.isComparing = true;
  }

  onRestartClicked() {
    this.isComparing = false;
    this.isComparingScenarios = false;
    this.slot1Visible = false;
    this.slot2Visible = false;
    this.restartFlag = true;
    setTimeout(() => this.restartFlag = false);
  }

  onSlotResult(event: { id: number, visible: boolean }) {
    if (event.id === 1) {
      this.slot1Visible = event.visible;
    } else if (event.id === 2) {
      this.slot2Visible = event.visible;
    }
  }

  onCompareScenarioClicked() {
    const slot1Data = this.slot1.getCurrentData();
    const slot2Data = this.slot2.getCurrentData();

    this.compareScenarioBody = {
      generatedTextSourcesFirstStory: slot1Data.generatedTextSources,
      generatedTextSourcesSecondStory: slot2Data.generatedTextSources,
      spinArgumentsFirstStory: slot1Data.spinArguments,
      spinArgumentsSecondStory: slot2Data.spinArguments
    }
    this.isComparingScenarios = true;
  }
}
