import { Footer, Layout, Navbar } from 'nextra-theme-docs'
import { Head } from 'nextra/components'
import { getPageMap } from 'nextra/page-map'
import 'nextra-theme-docs/style.css'

export const metadata = {
  title: {
    default: 'Orrery Documentation',
    template: '%s – Orrery'
  },
  description:
    'Orrery: interactive embedding visualisation with native SAE interpretability support.'
}

const navbar = (
  <Navbar
    logo={<b>Orrery</b>}
    projectLink="https://github.com/Giacomo-De-Luca/orrery"
  />
)

export default async function RootLayout({ children }) {
  return (
    <html lang="en" dir="ltr" suppressHydrationWarning>
      <Head />
      <body>
        <Layout
          navbar={navbar}
          pageMap={await getPageMap()}
          docsRepositoryBase="https://github.com/Giacomo-De-Luca/orrery/tree/main/docs"
          editLink={null}
          footer={<Footer>Apache 2.0 © Orrery</Footer>}
        >
          {children}
        </Layout>
      </body>
    </html>
  )
}
