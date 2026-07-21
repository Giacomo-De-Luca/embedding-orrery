/** Validates that separately fetched projections align with the loaded item core. */
export class ProjectionMembership {
  static assertCompatible(
    coreSignature: string | null | undefined,
    projectionSignature: string | null | undefined,
    projectionType: string,
  ): void {
    if (!coreSignature) {
      throw new Error(
        'loaded collection core is missing its item membership signature',
      );
    }
    if (!projectionSignature) {
      throw new Error(
        `${projectionType} response is missing its item membership signature`,
      );
    }

    if (coreSignature !== projectionSignature) {
      throw new Error(
        `${projectionType} item membership does not match the loaded collection core`,
      );
    }
  }
}
