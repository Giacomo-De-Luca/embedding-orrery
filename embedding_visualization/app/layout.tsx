import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";
import { Providers } from "./providers";
import {
  SITE_DESCRIPTION,
  SITE_NAME,
  SITE_TITLE,
  resolveSiteUrl,
} from "@/lib/utils/siteMetadata";

// Social preview: `app/opengraph-image.jpg` (+ `.alt.txt`) is picked up by the
// Next.js file convention and emitted as og:image / twitter:image. Those URLs
// are only absolute (crawler-fetchable) when metadataBase is set — see
// lib/utils/siteMetadata.ts for where NEXT_PUBLIC_SITE_URL comes from.
export const metadata: Metadata = {
  metadataBase: resolveSiteUrl(process.env.NEXT_PUBLIC_SITE_URL),
  title: SITE_TITLE,
  description: SITE_DESCRIPTION,
  openGraph: {
    type: "website",
    siteName: SITE_NAME,
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
  },
  twitter: {
    card: "summary_large_image",
  },
};

type RootLayoutProps = Readonly<{
  children: React.ReactNode;
}>;

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en" suppressHydrationWarning className={`${GeistSans.variable} ${GeistMono.variable}`}>
      <body className="antialiased">
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  );
}
