import { Component, HostListener, Input, OnInit } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
import { SpinService } from '../../services/spin.service';
import { CommonModule } from '@angular/common';
import { CompareScenariosRequest, GeneratedTextSources, SpinArguments } from '../../api/models';
import { Observable } from 'rxjs';
import { NgIconComponent, provideIcons } from '@ng-icons/core';
import { iconoirArrowUpRight, iconoirXmark } from "@ng-icons/iconoir";

@Component({
  selector: 'app-compare-scenario',
  standalone: true,
  imports: [CommonModule, NgIconComponent],
  templateUrl: './compare-scenario.component.html',
  styleUrls: ['./compare-scenario.component.css'],
  providers: [provideIcons({ iconoirArrowUpRight, iconoirXmark })]
})
export class CompareScenarioComponent implements OnInit {
  @Input() body: CompareScenariosRequest | undefined
  readonly selectGeneratedComparison$: Observable<GeneratedTextSources>
  isTooltipOpen = false;

  constructor(private spinService: SpinService) {
    this.selectGeneratedComparison$ = spinService.selectGeneratedComparison$;
  }

  ngOnInit() {
    if(this.body) this.spinService.streamCompareScenarios(this.body)
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
