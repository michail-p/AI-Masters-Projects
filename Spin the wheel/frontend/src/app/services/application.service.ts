import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ApplicationService {

  constructor() { }

  public effect<T>(
    serviceResult: Observable<T>,
    onSuccess: (r: T) => void,
    onError?: (e: unknown) => void,
    onFinally?: () => void
  ): void {
    serviceResult.subscribe({
      next: result => onSuccess(result),
      error: e => {
        console.error(e);
        if (onError) onError(e);
        if (onFinally) onFinally();
      },
      complete: () => {
        if (onFinally) onFinally();
      }
    });
  }
}
