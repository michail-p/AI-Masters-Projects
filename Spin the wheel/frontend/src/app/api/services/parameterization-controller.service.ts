/* tslint:disable */
/* eslint-disable */
import { HttpClient, HttpContext, HttpResponse } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { filter, map } from 'rxjs/operators';

import { BaseService } from '../base-service';
import { ApiConfiguration } from '../api-configuration';
import { StrictHttpResponse } from '../strict-http-response';
import { RequestBuilder } from '../request-builder';

import { Gender } from '../models/gender';

@Injectable({ providedIn: 'root' })
export class ParameterizationControllerService extends BaseService {
  constructor(config: ApiConfiguration, http: HttpClient) {
    super(config, http);
  }

  /** Path part for operation `getTimes()` */
  static readonly GetTimesPath = '/api/parameterization/times';

  /**
   * This method provides access to the full `HttpResponse`, allowing access to response headers.
   * To access only the response body, use `getTimes()` instead.
   *
   * This method doesn't expect any request body.
   */
  getTimes$Response(
    params?: {
    },
    context?: HttpContext
  ): Observable<StrictHttpResponse<Array<number>>> {
    const rb = new RequestBuilder(this.rootUrl, ParameterizationControllerService.GetTimesPath, 'get');
    if (params) {
    }

    return this.http.request(
      rb.build({ accept: 'application/json', context })
    ).pipe(
      filter((r: any): r is HttpResponse<any> => r instanceof HttpResponse),
      map((r: HttpResponse<any>) => {
        return r as StrictHttpResponse<Array<number>>;
      })
    );
  }

  /**
   * This method provides access only to the response body.
   * To access the full response (for headers, for example), `getTimes$Response()` instead.
   *
   * This method doesn't expect any request body.
   */
  getTimes(
    params?: {
    },
    context?: HttpContext
  ): Observable<Array<number>> {
    return this.getTimes$Response(params, context).pipe(
      map((r: StrictHttpResponse<Array<number>>): Array<number> => r.body)
    );
  }

  /** Path part for operation `getPlaces()` */
  static readonly GetPlacesPath = '/api/parameterization/places';

  /**
   * This method provides access to the full `HttpResponse`, allowing access to response headers.
   * To access only the response body, use `getPlaces()` instead.
   *
   * This method doesn't expect any request body.
   */
  getPlaces$Response(
    params?: {
    },
    context?: HttpContext
  ): Observable<StrictHttpResponse<Array<string>>> {
    const rb = new RequestBuilder(this.rootUrl, ParameterizationControllerService.GetPlacesPath, 'get');
    if (params) {
    }

    return this.http.request(
      rb.build({ accept: 'application/json', context })
    ).pipe(
      filter((r: any): r is HttpResponse<any> => r instanceof HttpResponse),
      map((r: HttpResponse<any>) => {
        return r as StrictHttpResponse<Array<string>>;
      })
    );
  }

  /**
   * This method provides access only to the response body.
   * To access the full response (for headers, for example), `getPlaces$Response()` instead.
   *
   * This method doesn't expect any request body.
   */
  getPlaces(
    params?: {
    },
    context?: HttpContext
  ): Observable<Array<string>> {
    return this.getPlaces$Response(params, context).pipe(
      map((r: StrictHttpResponse<Array<string>>): Array<string> => r.body)
    );
  }

  /** Path part for operation `getGenders()` */
  static readonly GetGendersPath = '/api/parameterization/genders';

  /**
   * This method provides access to the full `HttpResponse`, allowing access to response headers.
   * To access only the response body, use `getGenders()` instead.
   *
   * This method doesn't expect any request body.
   */
  getGenders$Response(
    params?: {
    },
    context?: HttpContext
  ): Observable<StrictHttpResponse<Array<Gender>>> {
    const rb = new RequestBuilder(this.rootUrl, ParameterizationControllerService.GetGendersPath, 'get');
    if (params) {
    }

    return this.http.request(
      rb.build({ accept: 'application/json', context })
    ).pipe(
      filter((r: any): r is HttpResponse<any> => r instanceof HttpResponse),
      map((r: HttpResponse<any>) => {
        return r as StrictHttpResponse<Array<Gender>>;
      })
    );
  }

  /**
   * This method provides access only to the response body.
   * To access the full response (for headers, for example), `getGenders$Response()` instead.
   *
   * This method doesn't expect any request body.
   */
  getGenders(
    params?: {
    },
    context?: HttpContext
  ): Observable<Array<Gender>> {
    return this.getGenders$Response(params, context).pipe(
      map((r: StrictHttpResponse<Array<Gender>>): Array<Gender> => r.body)
    );
  }

}
