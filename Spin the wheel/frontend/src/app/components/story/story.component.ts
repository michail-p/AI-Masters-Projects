import { AfterViewInit, ChangeDetectionStrategy, Component, HostListener, Input, OnInit } from '@angular/core';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';
import { SpinService } from '../../services/spin.service';
import { CommonModule } from '@angular/common';
import { GeneratedTextSources, SpinArguments } from '../../api/models';
import { Observable, map } from 'rxjs';
import { NgIconComponent, provideIcons } from '@ng-icons/core';
import { iconoirArrowUpRight, iconoirXmark } from "@ng-icons/iconoir";

@Component({
  selector: 'app-story',
  standalone: true,
  imports: [CommonModule, NgIconComponent],
  templateUrl: './story.component.html',
  styleUrls: ['./story.component.css'],
  changeDetection: ChangeDetectionStrategy.OnPush,
  providers: [provideIcons({ iconoirArrowUpRight, iconoirXmark })]
})
export class StoryComponent implements OnInit, AfterViewInit {
  @Input() body: SpinArguments | undefined;
  @Input() isComparing: boolean = false;
  readonly selectGeneratedText$: Observable<GeneratedTextSources>
  imageUrl$: Observable<SafeUrl | null> | null = null;
  isTooltipOpen = false;

  constructor(private spinService: SpinService, private sanitizer: DomSanitizer) {
    this.selectGeneratedText$ = spinService.selectGeneratedText$;
  }

  ngOnInit() {
    if (this.body) {
      this.imageUrl$ = this.spinService.selectGeneratedImage$
        .pipe(map(blob => this.sanitizer.bypassSecurityTrustUrl(URL.createObjectURL(blob))));
    }
  }

  ngAfterViewInit(): void {
    if (this.body) {
      // Trigger API calls after first change detection pass to avoid NG0100
      queueMicrotask(() => {
        this.spinService.getGeneratedImage(this.body!);
        this.spinService.streamGeneratedText(this.body!);
      });
    }
  }

  toggleTooltip() {
    this.isTooltipOpen = !this.isTooltipOpen;
  }

  formatSourceLabel(url: string): string {
    if (url.includes('skbl.se')) {
      return 'SKBL - Svenskt kvinnobiografiskt lexikon';
    }
    if (url.includes('huggingface.co')) {
      const match = url.match(/huggingface\.co\/([^\/]+\/[^\/]+)/);
      return match ? `HuggingFace: ${match[1]}` : 'HuggingFace Model';
    }
    try {
      return new URL(url).hostname;
    } catch {
      return url;
    }
  }

  @HostListener('document:click', ['$event'])
  onClick(event: Event) {
    if (!(event.target as HTMLElement).closest('.tooltip-icon') &&
      !(event.target as HTMLElement).closest('.tooltip-box')) {
      this.isTooltipOpen = false;
    }
  }

}
