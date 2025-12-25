import { Injectable } from "@angular/core";
import { BehaviorSubject, Observable } from "rxjs";

@Injectable({
  providedIn: 'root'
})
export class LoadingService {
  private loadingCount = 0;
  private loadingSubject = new BehaviorSubject<boolean>(false);
  public readonly loading$: Observable<boolean> = this.loadingSubject.asObservable();

  show() {
    if(this.loadingCount<0) this.loadingCount = 0;
    this.loadingCount++;
    if (this.loadingCount === 1) this.loadingSubject.next(true);
  }

  hide() {
    if(this.loadingCount<0) this.loadingCount = 0;
    if (this.loadingCount > 0) this.loadingCount--;
    if (this.loadingCount === 0) this.loadingSubject.next(false);
  }
}
