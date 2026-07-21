/** Monotonic token gate used to ignore stale asynchronous responses. */
export class LatestRequestGate {
  private latestToken = 0;

  begin(): number {
    this.latestToken += 1;
    return this.latestToken;
  }

  invalidate(): void {
    this.latestToken += 1;
  }

  isLatest(token: number): boolean {
    return token === this.latestToken;
  }
}
