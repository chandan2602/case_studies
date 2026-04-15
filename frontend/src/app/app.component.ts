import { Component, OnInit, Inject, PLATFORM_ID } from '@angular/core';
import { isPlatformBrowser, CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from './api.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
})
export class AppComponent implements OnInit {
  states: string[]    = [];
  districts: string[] = [];
  villages: string[]  = [];

  selectedState    = '';
  selectedDistrict = '';
  selectedVillage  = '';
  selectedFormat   = 'docx';

  loading = false;
  error   = '';

  get scopeLabel(): string {
    if (this.selectedVillage)  return `Village — ${this.selectedVillage}`;
    if (this.selectedDistrict) return `District — ${this.selectedDistrict}, ${this.selectedState}`;
    if (this.selectedState)    return `State — ${this.selectedState}`;
    return '';
  }

  constructor(
    private api: ApiService,
    @Inject(PLATFORM_ID) private platformId: object
  ) {}

  ngOnInit(): void {
    if (!isPlatformBrowser(this.platformId)) return;
    this.api.getStates().subscribe({
      next: r  => (this.states = r.states),
      error: () => (this.error = 'Could not connect to backend. Make sure it is running on port 8000.'),
    });
  }

  onStateChange(): void {
    this.districts = [];
    this.villages  = [];
    this.selectedDistrict = '';
    this.selectedVillage  = '';
    if (!this.selectedState) return;
    this.api.getDistricts(this.selectedState).subscribe(r => (this.districts = r.districts));
  }

  onDistrictChange(): void {
    this.villages = [];
    this.selectedVillage = '';
    if (!this.selectedDistrict) return;
    this.api
      .getVillages(this.selectedState, this.selectedDistrict)
      .subscribe(r => (this.villages = r.villages));
  }

  onSubmit(): void {
    if (!this.selectedState) return;
    this.loading = true;
    this.error   = '';

    this.api
      .generateReport(
        this.selectedState,
        this.selectedDistrict,
        this.selectedVillage,
        this.selectedFormat
      )
      .subscribe({
        next: (blob) => {
          const label = this.selectedVillage || this.selectedDistrict || this.selectedState;
          const safe  = label.replace(/[^\w]/g, '_');
          const url   = URL.createObjectURL(blob);
          const a     = document.createElement('a');
          a.href      = url;
          a.download  = `case_study_${safe}.${this.selectedFormat}`;
          a.click();
          URL.revokeObjectURL(url);
          this.loading = false;
        },
        error: (err) => {
          this.error   = err?.error?.error || 'Something went wrong. Please try again.';
          this.loading = false;
        },
      });
  }
}
