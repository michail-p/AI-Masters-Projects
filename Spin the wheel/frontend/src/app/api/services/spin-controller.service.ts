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

import { CompareScenariosRequest } from '../models/compare-scenarios-request';
import { GeneratedTextSources } from '../models/generated-text-sources';
import { SpinArguments } from '../models/spin-arguments';

@Injectable({ providedIn: 'root' })
export class SpinControllerService extends BaseService {
  constructor(config: ApiConfiguration, http: HttpClient) {
    super(config, http);
  }

  /** Path part for operation `postGeneratedText()` */
  static readonly PostGeneratedTextPath = '/api/spin/story';

  /**
   * This method provides access to the full `HttpResponse`, allowing access to response headers.
   * To access only the response body, use `postGeneratedText()` instead.
   *
   * This method sends `application/json` and handles request body of type `application/json`.
   */
  postGeneratedText$Response(
    params: {
      body: SpinArguments
    },
    context?: HttpContext
  ): Observable<StrictHttpResponse<GeneratedTextSources>> {
    const rb = new RequestBuilder(this.rootUrl, SpinControllerService.PostGeneratedTextPath, 'post');
    if (params) {
      rb.body(params.body, 'application/json');
    }

    return this.http.request(
      rb.build({ accept: 'application/json', context })
    ).pipe(
      filter((r: any): r is HttpResponse<any> => r instanceof HttpResponse),
      map((r: HttpResponse<any>) => {
        return r as StrictHttpResponse<GeneratedTextSources>;
      })
    );
  }

  /**
   * This method provides access only to the response body.
   * To access the full response (for headers, for example), `postGeneratedText$Response()` instead.
   *
   * This method sends `application/json` and handles request body of type `application/json`.
   */
  postGeneratedText(
    params: {
      body: SpinArguments
    },
    context?: HttpContext
  ): Observable<GeneratedTextSources> {
    return this.postGeneratedText$Response(params, context).pipe(
      map((r: StrictHttpResponse<GeneratedTextSources>): GeneratedTextSources => r.body)
    );
  }

  /** Path part for operation `postGeneratedImage()` */
  static readonly PostGeneratedImagePath = '/api/spin/image';

  /**
   * This method provides access to the full `HttpResponse`, allowing access to response headers.
   * To access only the response body, use `postGeneratedImage()` instead.
   *
   * This method sends `application/json` and handles request body of type `application/json`.
   */
  postGeneratedImage$Response(
    params: {
      body: SpinArguments
    },
    context?: HttpContext
  ): Observable<StrictHttpResponse<Blob>> {
    const rb = new RequestBuilder(this.rootUrl, SpinControllerService.PostGeneratedImagePath, 'post');
    if (params) {
      rb.body(params.body, 'application/json');
    }

    return this.http.request(
      rb.build({ responseType: 'blob', accept: '*/*', context })
    ).pipe(
      filter((r: any): r is HttpResponse<any> => r instanceof HttpResponse),
      map((r: HttpResponse<any>) => {
        return r as StrictHttpResponse<Blob>;
      })
    );
  }

  /**
   * This method provides access only to the response body.
   * To access the full response (for headers, for example), `postGeneratedImage$Response()` instead.
   *
   * This method sends `application/json` and handles request body of type `application/json`.
   */
  postGeneratedImage(
    params: {
      body: SpinArguments
    },
    context?: HttpContext
  ): Observable<Blob> {
    return this.postGeneratedImage$Response(params, context).pipe(
      map((r: StrictHttpResponse<Blob>): Blob => r.body)
    );
  }

  /** Path part for operation `postCompareScenarios()` */
  static readonly PostCompareScenariosPath = '/api/spin/compare-scenarios';

  /**
   * This method provides access to the full `HttpResponse`, allowing access to response headers.
   * To access only the response body, use `postCompareScenarios()` instead.
   *
   * This method sends `application/json` and handles request body of type `application/json`.
   */
  postCompareScenarios$Response(
    params: {
      body: CompareScenariosRequest
    },
    context?: HttpContext
  ): Observable<StrictHttpResponse<GeneratedTextSources>> {
    const rb = new RequestBuilder(this.rootUrl, SpinControllerService.PostCompareScenariosPath, 'post');
    if (params) {
      rb.body(params.body, 'application/json');
    }

    return this.http.request(
      rb.build({ accept: 'application/json', context })
    ).pipe(
      filter((r: any): r is HttpResponse<any> => r instanceof HttpResponse),
      map((r: HttpResponse<any>) => {
        return r as StrictHttpResponse<GeneratedTextSources>;
      })
    );
  }

  /**
   * This method provides access only to the response body.
   * To access the full response (for headers, for example), `postCompareScenarios$Response()` instead.
   *
   * This method sends `application/json` and handles request body of type `application/json`.
   */
  postCompareScenarios(
    params: {
      body: CompareScenariosRequest
    },
    context?: HttpContext
  ): Observable<GeneratedTextSources> {
    return this.postCompareScenarios$Response(params, context).pipe(
      map((r: StrictHttpResponse<GeneratedTextSources>): GeneratedTextSources => r.body)
    );
  }

}
