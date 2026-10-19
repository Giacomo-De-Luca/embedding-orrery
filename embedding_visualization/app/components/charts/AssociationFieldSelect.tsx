'use client';

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import type { AssociationFieldOption } from '@/lib/utils/fieldAxis';
import { cn } from '@/lib/utils/utils';

interface AssociationFieldSelectProps {
  label: string;
  value: string | null;
  options: readonly AssociationFieldOption[];
  onChange: (field: string) => void;
  disabled?: boolean;
  /** `stacked`: label above (sidebar); `inline`: label inside the trigger (dialog toolbar). */
  variant?: 'stacked' | 'inline';
  className?: string;
}

/** Field picker of the association view: categorical fields, then numeric ones (binned). */
export function AssociationFieldSelect({
  label,
  value,
  options,
  onChange,
  disabled,
  variant = 'stacked',
  className,
}: AssociationFieldSelectProps) {
  const categorical = options.filter(o => o.kind === 'categorical');
  const numeric = options.filter(o => o.kind === 'numeric');

  const select = (
    <Select value={value ?? ''} onValueChange={onChange} disabled={disabled}>
      <SelectTrigger
        size="sm"
        className={cn('h-7 text-xs', variant === 'stacked' && 'w-full')}
        aria-label={variant === 'inline' ? label : undefined}
      >
        {variant === 'inline' && <span className="text-muted-foreground">{label}:</span>}
        <SelectValue placeholder="Select field..." />
      </SelectTrigger>
      <SelectContent>
        {categorical.length > 0 && (
          <SelectGroup>
            <SelectLabel className="text-[10px]">Categorical</SelectLabel>
            {categorical.map(opt => (
              <SelectItem key={opt.field} value={opt.field}>
                <span className="text-xs">{opt.displayName} ({opt.uniqueCount})</span>
              </SelectItem>
            ))}
          </SelectGroup>
        )}
        {numeric.length > 0 && (
          <SelectGroup>
            <SelectLabel className="text-[10px]">Numeric (binned)</SelectLabel>
            {numeric.map(opt => (
              <SelectItem key={opt.field} value={opt.field}>
                <span className="text-xs">{opt.displayName}</span>
              </SelectItem>
            ))}
          </SelectGroup>
        )}
      </SelectContent>
    </Select>
  );

  if (variant === 'inline') return <div className={className}>{select}</div>;
  return (
    <label className={cn('flex min-w-0 flex-1 flex-col gap-1', className)}>
      <span className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</span>
      {select}
    </label>
  );
}
