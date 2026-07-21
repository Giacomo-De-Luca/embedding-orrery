import { describe, expect, it } from 'vitest';

import { LatestRequestGate } from '../latestRequestGate';

describe('LatestRequestGate', () => {
  it('accepts only the newest request token', () => {
    const gate = new LatestRequestGate();
    const first = gate.begin();
    const second = gate.begin();

    expect(gate.isLatest(first)).toBe(false);
    expect(gate.isLatest(second)).toBe(true);
  });

  it('invalidates an in-flight token without starting network work', () => {
    const gate = new LatestRequestGate();
    const request = gate.begin();

    gate.invalidate();

    expect(gate.isLatest(request)).toBe(false);
  });
});
