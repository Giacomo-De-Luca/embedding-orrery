import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  getOnboardingAction,
  planOnboarding,
  shouldShowMobileNotice,
  tourFitsViewport,
  collectionFitsViewport,
  linkedCollectionsFitViewport,
  warmEmotionSearch,
  resetWarmEmotionSearchForTests,
  INTRO_STORAGE_KEY,
  TOUR_STORAGE_KEY,
  SAE_TOUR_STORAGE_KEY,
  MOBILE_NOTICE_STORAGE_KEY,
  COMPACT_VIEWPORT_WIDTH,
} from '../demoOnboarding';
import { TOUR_COLLECTION, WORDNET_COLLECTION } from '../tourPresets';

const base = {
  isDemo: true,
  search: '',
  introSeen: false,
  viewportWidth: 1200,
  mobileNoticeSeen: false,
};

/** A phone-sized viewport that has already dismissed the desktop notice. */
const narrowSeen = { viewportWidth: COMPACT_VIEWPORT_WIDTH - 1, mobileNoticeSeen: true };

describe('getOnboardingAction', () => {
  it('auto-opens the intro on a bare demo first visit', () => {
    expect(getOnboardingAction(base)).toBe('intro');
  });

  it('never auto-opens outside demo builds or after being seen', () => {
    expect(getOnboardingAction({ ...base, isDemo: false })).toBeNull();
    expect(getOnboardingAction({ ...base, introSeen: true })).toBeNull();
  });

  it.each(['?collection=emotion', '?colorBy=label', '?preset=emnlp-topics'])(
    'deep link %s suppresses the auto-intro',
    (search) => {
      expect(getOnboardingAction({ ...base, search })).toBeNull();
    },
  );

  it('?intro=1 opens the dialog in any build, ignoring storage', () => {
    expect(getOnboardingAction({ ...base, search: '?intro=1', isDemo: false, introSeen: true }))
      .toBe('intro');
  });

  it('?tour=1 starts the tour in any build', () => {
    expect(getOnboardingAction({ ...base, search: '?tour=1', isDemo: false, introSeen: true }))
      .toBe('tour');
  });

  it('?tour=1 runs on phones too', () => {
    expect(getOnboardingAction({ ...base, ...narrowSeen, search: '?tour=1' })).toBe('tour');
  });

  it('?tour=1 wins over ?intro=1', () => {
    expect(getOnboardingAction({ ...base, search: '?tour=1&intro=1' })).toBe('tour');
  });

  it('?tour=sae routes to the SAE tour in any build', () => {
    expect(getOnboardingAction({ ...base, search: '?tour=sae' })).toBe('sae-tour');
    expect(
      getOnboardingAction({ ...base, search: '?tour=sae', isDemo: false, introSeen: true }),
    ).toBe('sae-tour');
  });

  it('?tour=sae wins over ?intro=1', () => {
    expect(getOnboardingAction({ ...base, search: '?tour=sae&intro=1' })).toBe('sae-tour');
  });

  it('?tour=wordnet / ?tour=probe route to their tours in any build', () => {
    expect(getOnboardingAction({ ...base, search: '?tour=wordnet' })).toBe('wordnet-tour');
    expect(getOnboardingAction({ ...base, search: '?tour=probe' })).toBe('probe-tour');
    expect(
      getOnboardingAction({ ...base, search: '?tour=wordnet', isDemo: false, introSeen: true }),
    ).toBe('wordnet-tour');
  });

  it.each(['?tour=sae', '?tour=probe', '?tour=wordnet'])(
    '%s is desktop-only: a phone gets the welcome dialog instead',
    (search) => {
      expect(getOnboardingAction({ ...base, ...narrowSeen, search })).toBe('intro');
      // Even in a non-demo build with the intro already seen: the link asked for a tour.
      expect(
        getOnboardingAction({ ...base, ...narrowSeen, search, isDemo: false, introSeen: true }),
      ).toBe('intro');
    },
  );

  it.each(['?tour=nope', '?tour=constructor', '?tour=__proto__'])(
    'unknown %s falls through to the ordinary gating',
    (search) => {
      expect(getOnboardingAction({ ...base, search })).toBeNull();
    },
  );

  it('shows the mobile notice on a phone-sized demo first visit', () => {
    expect(
      getOnboardingAction({ ...base, viewportWidth: COMPACT_VIEWPORT_WIDTH - 1 }),
    ).toBe('mobile-notice');
  });

  it('the phone-fitting ?tour=1 wins over the mobile notice', () => {
    expect(
      getOnboardingAction({ ...base, search: '?tour=1', viewportWidth: COMPACT_VIEWPORT_WIDTH - 1 }),
    ).toBe('tour');
  });

  it.each(['?intro=1', '?tour=sae', '?tour=probe', '?tour=wordnet'])(
    'the mobile notice pre-empts %s',
    (search) => {
      expect(
        getOnboardingAction({ ...base, search, viewportWidth: COMPACT_VIEWPORT_WIDTH - 1 }),
      ).toBe('mobile-notice');
    },
  );

  it('once dismissed, narrow viewports resume the ordinary gating', () => {
    expect(getOnboardingAction({ ...base, ...narrowSeen })).toBe('intro');
    expect(getOnboardingAction({ ...base, ...narrowSeen, introSeen: true })).toBeNull();
  });
});

describe('planOnboarding', () => {
  const phone = { ...base, viewportWidth: 390 };

  it('follows the mobile notice with the welcome dialog it pre-empted', () => {
    expect(planOnboarding(phone)).toEqual({ action: 'mobile-notice', afterNotice: 'intro' });
    expect(planOnboarding({ ...phone, search: '?intro=1', introSeen: true })).toEqual({
      action: 'mobile-notice',
      afterNotice: 'intro',
    });
    // A desktop-only tour link downgrades to the dialog once the notice is gone.
    expect(planOnboarding({ ...phone, search: '?tour=wordnet' })).toEqual({
      action: 'mobile-notice',
      afterNotice: 'intro',
    });
  });

  it('follows the notice with nothing when no dialog was due', () => {
    expect(planOnboarding({ ...phone, introSeen: true })).toEqual({
      action: 'mobile-notice',
      afterNotice: null,
    });
    expect(planOnboarding({ ...phone, search: '?collection=emotion' })).toEqual({
      action: 'mobile-notice',
      afterNotice: null,
    });
  });

  it('has no follow-up when the notice does not show', () => {
    expect(planOnboarding(base)).toEqual({ action: 'intro', afterNotice: null });
    expect(planOnboarding({ ...phone, search: '?tour=1' })).toEqual({
      action: 'tour',
      afterNotice: null,
    });
  });
});

describe('tourFitsViewport', () => {
  it('runs every tour on a desktop', () => {
    for (const tour of ['tour', 'sae-tour', 'wordnet-tour', 'probe-tour'] as const) {
      expect(tourFitsViewport(tour, COMPACT_VIEWPORT_WIDTH)).toBe(true);
    }
  });

  it('runs only the 90-second tour on phones', () => {
    expect(tourFitsViewport('tour', 390)).toBe(true);
    for (const tour of ['sae-tour', 'wordnet-tour', 'probe-tour'] as const) {
      expect(tourFitsViewport(tour, COMPACT_VIEWPORT_WIDTH - 1)).toBe(false);
    }
  });
});

describe('collectionFitsViewport', () => {
  const phone = { isDemo: true, viewportWidth: COMPACT_VIEWPORT_WIDTH - 1 };

  it('drops WordNet on demo phones only', () => {
    expect(collectionFitsViewport(WORDNET_COLLECTION, phone)).toBe(false);
    expect(
      collectionFitsViewport(WORDNET_COLLECTION, { ...phone, viewportWidth: COMPACT_VIEWPORT_WIDTH }),
    ).toBe(true);
    // A self-hoster on a phone keeps every collection.
    expect(collectionFitsViewport(WORDNET_COLLECTION, { ...phone, isDemo: false })).toBe(true);
  });

  it('keeps every other collection on demo phones', () => {
    expect(collectionFitsViewport(TOUR_COLLECTION, phone)).toBe(true);
    expect(collectionFitsViewport('emotion', phone)).toBe(true);
  });
});

describe('linkedCollectionsFitViewport', () => {
  const phone = { isDemo: true, viewportWidth: COMPACT_VIEWPORT_WIDTH - 1 };

  it('rejects a link naming WordNet on a demo phone, by URL or by preset', () => {
    expect(linkedCollectionsFitViewport([WORDNET_COLLECTION, null], phone)).toBe(false);
    expect(linkedCollectionsFitViewport([null, WORDNET_COLLECTION], phone)).toBe(false);
    expect(linkedCollectionsFitViewport(['emotion', WORDNET_COLLECTION], phone)).toBe(false);
  });

  it('accepts links without a desktop-only collection, and every link on desktop', () => {
    expect(linkedCollectionsFitViewport([null, undefined], phone)).toBe(true);
    expect(linkedCollectionsFitViewport(['emotion', TOUR_COLLECTION], phone)).toBe(true);
    expect(
      linkedCollectionsFitViewport([WORDNET_COLLECTION], { isDemo: true, viewportWidth: 1200 }),
    ).toBe(true);
  });
});

describe('shouldShowMobileNotice', () => {
  const narrow = { isDemo: true, viewportWidth: 390, mobileNoticeSeen: false };

  it('fires on an unseen phone-sized demo viewport', () => {
    expect(shouldShowMobileNotice(narrow)).toBe(true);
  });

  it('is demo-only — self-hosted builds are never gated', () => {
    expect(shouldShowMobileNotice({ ...narrow, isDemo: false })).toBe(false);
  });

  it('does not fire at or above the viewport floor', () => {
    expect(shouldShowMobileNotice({ ...narrow, viewportWidth: COMPACT_VIEWPORT_WIDTH })).toBe(false);
    expect(shouldShowMobileNotice({ ...narrow, viewportWidth: COMPACT_VIEWPORT_WIDTH - 1 })).toBe(true);
  });

  it('does not fire once dismissed', () => {
    expect(shouldShowMobileNotice({ ...narrow, mobileNoticeSeen: true })).toBe(false);
  });
});

describe('storage keys', () => {
  it('are the versioned constants', () => {
    expect(INTRO_STORAGE_KEY).toBe('orrery.demo-intro.v1');
    expect(TOUR_STORAGE_KEY).toBe('orrery.demo-tour.v1');
    expect(SAE_TOUR_STORAGE_KEY).toBe('orrery.demo-sae-tour.v1');
    expect(MOBILE_NOTICE_STORAGE_KEY).toBe('orrery.demo-mobile-notice.v1');
  });
});

describe('warmEmotionSearch', () => {
  beforeEach(resetWarmEmotionSearchForTests);

  it('no-ops outside demo builds', () => {
    const query = vi.fn().mockResolvedValue({});
    warmEmotionSearch({ query }, false);
    expect(query).not.toHaveBeenCalled();
  });

  it('fires once against emotion only, then never again', () => {
    const query = vi.fn().mockResolvedValue({});
    warmEmotionSearch({ query }, true);
    warmEmotionSearch({ query }, true);
    expect(query).toHaveBeenCalledTimes(1);
    const { variables, fetchPolicy } = query.mock.calls[0][0];
    expect(variables.collectionName).toBe('emotion');
    expect(variables.nResults).toBe(1);
    expect(fetchPolicy).toBe('no-cache');
  });
});
