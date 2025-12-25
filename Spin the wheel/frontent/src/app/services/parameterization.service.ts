import { Injectable } from '@angular/core';
import { BehaviorSubject, filter, map, Observable } from 'rxjs';
import { ApplicationService } from './application.service';
import { ParameterizationControllerService } from '../api/services';
import { Gender } from '../api/models';
import { LoadingService } from './loading.service';

interface ParameterizationState {
  places?: string[],
  times?: number[],
  genders?: Gender[]
}

@Injectable({
  providedIn: 'root'
})
export class ParameterizationService {
  private readonly state$: BehaviorSubject<ParameterizationState> = new BehaviorSubject<ParameterizationState>({});
  readonly selectPlaces$: Observable<string[]> = this.state$.pipe(map(state => state.places)).pipe(filter((p) => !!p), map((p) => p!));
  readonly selectTimes$: Observable<number[]> = this.state$.pipe(map(state => state.times)).pipe(filter((p) => !!p), map((p) => p!));
  readonly selectGenders$: Observable<Gender[]> = this.state$.pipe(map(state => state.genders)).pipe(filter((p) => !!p), map((p) => p!));

  constructor(private applicationService: ApplicationService, private rest: ParameterizationControllerService, private loadingService: LoadingService) { }

  getPlaces() {
    this.loadingService.show();
    this.applicationService.effect(
      this.rest.getPlaces(),
      (places) => { this.updateState((s) => ({ ...s, places })); },
      undefined,
      () => this.loadingService.hide()
    );
  }

  getTimes() {
    this.loadingService.show();
    this.applicationService.effect(
      this.rest.getTimes(),
      (times) => { this.updateState((s) => ({ ...s, times })); },
      undefined,
      () => this.loadingService.hide()
    );
  }

  getGenders() {
    this.loadingService.show();
    this.applicationService.effect(
      this.rest.getGenders(),
      (genders) => { this.updateState((s) => ({ ...s, genders })); },
      undefined,
      () => this.loadingService.hide()
    );
  }

  cleanState() { this.state$.next({}); }

  updateState(nextState: (previous: ParameterizationState) => ParameterizationState) {
    this.state$.next(nextState(this.state$.getValue()));
  }
}