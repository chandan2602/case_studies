import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

const BASE = 'http://localhost:8000';

@Injectable({ providedIn: 'root' })
export class ApiService {
  constructor(private http: HttpClient) {}

  getStates(): Observable<{ states: string[] }> {
    return this.http.get<{ states: string[] }>(`${BASE}/get_states`);
  }

  getDistricts(state: string): Observable<{ districts: string[] }> {
    return this.http.get<{ districts: string[] }>(`${BASE}/get_districts`, {
      params: { state },
    });
  }

  getVillages(state: string, district: string): Observable<{ villages: string[] }> {
    return this.http.get<{ villages: string[] }>(`${BASE}/get_villages`, {
      params: { state, district },
    });
  }

  generateReport(
    state: string,
    district: string,
    village: string,
    format: string
  ): Observable<Blob> {
    return this.http.post(
      `${BASE}/generate_case_study`,
      { state, district, village, format },
      { responseType: 'blob' }
    );
  }
}
