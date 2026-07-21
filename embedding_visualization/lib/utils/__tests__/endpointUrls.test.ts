import { describe, expect, it } from 'vitest';

import {
  resolveApiBaseUrl,
  resolveGraphqlHttpUrl,
  resolveGraphqlWebSocketUrl,
} from '../endpointUrls';

describe('container endpoint URL resolution', () => {
  it('uses the local backend when public configuration is absent', () => {
    expect(resolveGraphqlHttpUrl(undefined)).toBe('http://localhost:8000/graphql');
    expect(resolveApiBaseUrl(undefined)).toBe('http://localhost:8000');
    expect(
      resolveGraphqlWebSocketUrl(undefined, {
        protocol: 'http:',
        host: 'localhost:3000',
      }),
    ).toBe('ws://localhost:8000/graphql');
  });

  it('uses same-origin endpoints for explicitly empty container configuration', () => {
    expect(resolveGraphqlHttpUrl('')).toBe('/graphql');
    expect(resolveGraphqlHttpUrl('/graphql')).toBe('/graphql');
    expect(resolveApiBaseUrl('')).toBe('');
    expect(
      resolveGraphqlWebSocketUrl('', {
        protocol: 'https:',
        host: 'orrery.example',
      }),
    ).toBe('wss://orrery.example/graphql');
  });

  it('preserves explicit development overrides', () => {
    expect(resolveGraphqlHttpUrl('http://localhost:8000/graphql')).toBe(
      'http://localhost:8000/graphql',
    );
    expect(
      resolveGraphqlWebSocketUrl('ws://localhost:8000/graphql', {
        protocol: 'https:',
        host: 'ignored.example',
      }),
    ).toBe('ws://localhost:8000/graphql');
    expect(resolveApiBaseUrl('http://localhost:8000/')).toBe('http://localhost:8000');
  });
});
