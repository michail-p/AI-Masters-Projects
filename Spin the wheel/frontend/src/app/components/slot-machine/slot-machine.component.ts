import { Component, Input, OnInit, OnChanges, SimpleChanges, Output, EventEmitter } from '@angular/core';
import { Gender, GeneratedTextSources, SpinArguments } from '../../api/models';
import { CommonModule } from '@angular/common';
import { StoryComponent } from '../story/story.component';
import { SpinService } from '../../services/spin.service';
import { LoadingService } from '../../services/loading.service';

@Component({
  selector: 'app-slot-machine',
  standalone: true,
  imports: [CommonModule, StoryComponent],
  templateUrl: './slot-machine.component.html',
  styleUrls: ['./slot-machine.component.css'],
  // Provide per-slot instances so one slot's loading state or generated text does not affect the other.
  providers: [SpinService, LoadingService]
})
export class SlotMachineComponent implements OnInit, OnChanges {
  @Input() id!: number;
  @Input() places: string[] | null | undefined;
  @Input() times: number[] | null | undefined;
  @Input() genders: Gender[] | null | undefined;
  @Input() isComparing: boolean = false;
  @Input() restart: boolean = false;
  @Output() resultVisibleChange = new EventEmitter<{ id: number, visible: boolean }>();
  body: SpinArguments = {};
  slots: any[] = [];
  spinning = false;
  resultVisible = false;
  resultImage = '';
  resultText = '';
  canSpin = false;

  constructor(private spinService: SpinService) { }

  ngOnInit() {
    this.slots = [
      { list: [], transform: 'translateY(0px)', duration: 0, items: this.places },
      { list: [], transform: 'translateY(0px)', duration: 0, items: this.times },
      { list: [], transform: 'translateY(0px)', duration: 0, items: this.genders }
    ];
    this.updateCanSpin();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (this.slots.length) {
      this.slots[0].items = this.places ?? [];
      this.slots[1].items = this.times ?? [];
      this.slots[2].items = this.genders ?? [];
      this.updateCanSpin();
    }
    if (changes['restart'] && changes['restart'].currentValue === true) {
      this.resetComponent();
    }
  }

  spin() {
    if (this.spinning || !this.canSpin) return;
    this.resultVisible = false;
    this.spinning = true;
    this.slots.forEach((slot: any, index) => {
      const items = Array.isArray(slot.items) ? slot.items : [];

      if (items.length === 0) return;

      slot.list = [...items, ...items, ...items];

      const finalIndex = Math.floor(Math.random() * items.length);
      const spins = 3;
      const itemHeight = 60;
      const translate = -(spins * items.length + finalIndex) * itemHeight;

      slot.duration = 1000 + index * 300;
      slot.transform = `translateY(${translate}px)`;
      setTimeout(() => {
        slot.list = [items[finalIndex]];
        slot.transform = 'translateY(0px)';

        if (index === this.slots.length - 1) {
          this.body.city = this.slots[0].list[0]
          this.body.year = this.slots[1].list[0]
          this.body.gender = this.slots[2].list[0]
          this.resultVisible = true;
          this.spinning = false;
          this.resultVisibleChange.emit({
            id: this.id,
            visible: true
          });
        }
      }, slot.duration);
    });
  }

  formatItem(item: any) {
    if (typeof item === 'string' || typeof item === 'number') return item;
    if (item && item.description) return item.description;
    return '';
  }

  resetComponent() {
    this.spinning = false;
    this.resultVisible = false;
    this.resultImage = '';
    this.resultText = '';
    this.body = {};

    this.slots = [
      { list: [], transform: 'translateY(0px)', duration: 0, items: this.places },
      { list: [], transform: 'translateY(0px)', duration: 0, items: this.times },
      { list: [], transform: 'translateY(0px)', duration: 0, items: this.genders }
    ];

    this.updateCanSpin();

    this.resultVisibleChange.emit({ id: this.id, visible: false });
  }

  getCurrentData(): { id: number, spinArguments: SpinArguments, generatedTextSources?: GeneratedTextSources } {
    return {
      id: this.id,
      spinArguments: {
        city: this.body.city,
        gender: this.body.gender,
        year: this.body.year
      },
      generatedTextSources: this.spinService.getGeneratedTextInState()
    };
  }

  private updateCanSpin() {
    const havePlaces = Array.isArray(this.places) && this.places.length > 0;
    const haveTimes = Array.isArray(this.times) && this.times.length > 0;
    const haveGenders = Array.isArray(this.genders) && this.genders.length > 0;
    this.canSpin = havePlaces && haveTimes && haveGenders;
  }
}
