import { ApplicationConfig, provideZoneChangeDetection } from '@angular/core';
import { provideRouter, withHashLocation } from '@angular/router';

import { routes } from './app.routes';
import { HTTP_INTERCEPTORS, provideHttpClient, withInterceptorsFromDi } from '@angular/common/http';
import { ApiModule } from './api/api.module';
import { ApiConfiguration } from './api/api-configuration';
import { provideAnimations } from '@angular/platform-browser/animations';

export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(routes, withHashLocation()),
    provideHttpClient(withInterceptorsFromDi()),
    ApiModule, { provide: ApiConfiguration, useValue: { rootUrl: 'http://localhost:8080' } },
    provideAnimations(),
    provideZoneChangeDetection({ eventCoalescing: true })
  ]
};
