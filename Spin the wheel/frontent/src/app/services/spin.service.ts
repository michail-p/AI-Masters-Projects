import { Injectable } from '@angular/core';
import { BehaviorSubject, filter, map, Observable } from 'rxjs';
import { ApplicationService } from './application.service';
import { SpinControllerService } from '../api/services';
import { CompareScenariosRequest, GeneratedTextSources, SpinArguments } from '../api/models';
import { LoadingService } from './loading.service';
import { ApiConfiguration } from '../api/api-configuration';

interface SpinState {
  generatedText?: GeneratedTextSources,
  compareScenariosText?: GeneratedTextSources,
  generatedImage?: Blob,
}

@Injectable({
  providedIn: 'root'
})
export class SpinService {
  private readonly state$: BehaviorSubject<SpinState> = new BehaviorSubject<SpinState>({});
  readonly selectGeneratedText$: Observable<GeneratedTextSources> = this.state$.pipe(map(state => state.generatedText)).pipe(filter((p) => !!p), map((p) => p!));
  readonly selectGeneratedComparison$: Observable<GeneratedTextSources> = this.state$.pipe(map(state => state.compareScenariosText)).pipe(filter((p) => !!p), map((p) => p!));
  readonly selectGeneratedImage$: Observable<Blob> = this.state$.pipe(map(state => state.generatedImage)).pipe(filter((p) => !!p), map((p) => p!));

  constructor(
    private applicationService: ApplicationService,
    private rest: SpinControllerService,
    private loadingService: LoadingService,
    private apiConfiguration: ApiConfiguration,
  ) { }

  getGeneratedImage(args: SpinArguments) {
    this.loadingService.show();
    this.applicationService.effect(
      this.rest.postGeneratedImage({ body: args }),
      (blob) => {
        this.updateState(s => ({ ...s, generatedImage: blob }));
      },
      undefined,
      () => this.loadingService.hide()
    );
  }

  streamGeneratedText(body: SpinArguments) {
    const url = `${this.apiConfiguration.rootUrl}/api/spin/story/stream`;
    this.cleanState();
    let sources: string[] = [];
    this.streamSse(url, body, (delta) => {
      this.updateState((s) => {
        const existing = s.generatedText?.generatedText ?? '';
        return {
          ...s,
          generatedText: {
            generatedText: existing + delta,
            sources: sources
          }
        };
      });
    }, (receivedSources) => {
      sources = receivedSources;
    });
  }

  streamCompareScenarios(body: CompareScenariosRequest) {
    const url = `${this.apiConfiguration.rootUrl}/api/spin/compare-scenarios/stream`;

    // Reset only comparison text to allow stories to remain visible.
    this.updateState((s) => ({ ...s, compareScenariosText: undefined }));

    let sources: string[] = [];
    this.streamSse(url, body, (delta) => {
      this.updateState((s) => {
        const existing = s.compareScenariosText?.generatedText ?? '';
        return {
          ...s,
          compareScenariosText: {
            generatedText: existing + delta,
            sources: sources
          }
        };
      });
    }, (receivedSources) => {
      sources = receivedSources;
    });
  }

  private async streamSse(url: string, body: unknown, onDelta: (delta: string) => void, onSources?: (sources: string[]) => void) {
    const decoder = new TextDecoder();
    this.loadingService.show();

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      if (!response.ok || !response.body) {
        throw new Error(`Streaming request failed (${response.status})`);
      }

      const reader = response.body.getReader();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() ?? '';

        for (const event of events) {
          const lines = event.split(/\r?\n/);
          
          // Check for event type (sources event)
          let eventType = 'message';
          for (const rawLine of lines) {
            const line = rawLine.trimEnd();
            if (line.startsWith('event:')) {
              eventType = line.substring(6).trim();
            }
          }
          
          for (const rawLine of lines) {
            const line = rawLine.trimEnd();
            if (!line.startsWith('data:')) continue;

            let data = line.substring(5);

            // Remove repeated data: prefixes on this line only, preserving any spaces in the payload
            while (data.startsWith('data:') || data.startsWith(' data:')) {
              data = data.replace(/^\s*data:/i, '');
            }

            if (!data || data === '[DONE]') continue;

            // Handle sources event
            if (eventType === 'sources' && onSources) {
              const sources = data.split(',').map(s => s.trim()).filter(s => s);
              onSources(sources);
              continue;
            }

            try {
              const isJson = data.trim().startsWith('{');
              const delta = isJson ? this.extractDelta(JSON.parse(data)) : data;
              if (delta) onDelta(delta);
            } catch (err) {
              console.error('Stream parse error', err);
            }
          }
        }
      }
    } catch (err) {
      console.error('Streaming failed', err);
    } finally {
      this.loadingService.hide();
    }
  }

  private extractDelta(json: any): string | undefined {
    const choices = json?.choices;
    if (!Array.isArray(choices) || choices.length === 0) return undefined;

    const delta = choices[0]?.delta;
    if (!delta) return undefined;

    const content = delta.content;

    if (typeof content === 'string') return content;

    if (Array.isArray(content)) {
      let text = '';
      for (const c of content) {
        if (typeof c?.text === 'string') text += c.text;
      }
      return text || undefined;
    }

    return undefined;
  }

  cleanState() { this.state$.next({}); }

  updateState(nextState: (previous: SpinState) => SpinState) {
    this.state$.next(nextState(this.state$.getValue()));
  }

  getGeneratedTextInState(): GeneratedTextSources | undefined { return this.state$.getValue().generatedText; }
}