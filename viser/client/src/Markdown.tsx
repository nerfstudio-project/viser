import * as runtime from 'react/jsx-runtime'
import * as provider from '@mdx-js/react'
import { evaluate } from '@mdx-js/mdx'
import { ReactNode, useEffect, useState } from 'react'

const components = {/* … */ }

async function parseMarkdown(markdown: string) {
    // @ts-ignore (necessary since JSX runtime isn't properly typed according to the internet)
    const { default: Content } = await evaluate(markdown, { ...runtime, ...provider, development: false })
    return Content;
}

export default function Markdown(props: { children?: string }) {
    const [child, setChild] = useState<ReactNode>(null)

    useEffect(() => {
        parseMarkdown(props.children ?? '').then((Content) => {
            setChild((<Content components={components} />))
        })
    }, [])

    return child;
}
