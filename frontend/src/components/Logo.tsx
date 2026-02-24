import { motion, useAnimation } from 'framer-motion'
import { useEffect } from 'react'

interface LogoProps {
  onClick?: () => void
  size?: 'default' | 'large'
}

const BAR_CONFIGS = [
  { baseH: 8,  x: 6,  animH: [8,  18, 10, 22, 8],  duration: 1.8, delay: 0.0  },
  { baseH: 16, x: 12, animH: [16, 8,  24, 12, 16], duration: 2.1, delay: 0.15 },
  { baseH: 24, x: 18, animH: [24, 14, 28, 10, 24], duration: 1.6, delay: 0.3  },
  { baseH: 14, x: 24, animH: [14, 26, 8,  20, 14], duration: 2.3, delay: 0.1  },
  { baseH: 10, x: 30, animH: [10, 20, 14, 6,  10], duration: 1.9, delay: 0.25 },
]

const BAR_WIDTH = 4
const BAR_Y_BASE = 34 // bottom anchor in the 40×40 viewBox

const SIZE_CONFIG = {
  default: { iconPx: 40, gap: 'gap-3', titleCls: 'text-base', subtitleCls: 'text-xs' },
  large:   { iconPx: 56, gap: 'gap-4', titleCls: 'text-xl',   subtitleCls: 'text-sm' },
}

export function Logo({ onClick, size = 'default' }: LogoProps) {
  const controls = useAnimation()
  const { iconPx, gap, titleCls, subtitleCls } = SIZE_CONFIG[size]
  const isInteractive = !!onClick

  useEffect(() => {
    controls.start('visible')
  }, [controls])

  const containerProps = {
    className: `flex items-center ${gap} select-none group ${
      isInteractive
        ? 'bg-transparent border-0 p-0 cursor-pointer'
        : 'cursor-default'
    }`,
    initial: 'hidden',
    animate: controls,
    variants: {
      hidden: { opacity: 0, x: -12 },
      visible: { opacity: 1, x: 0, transition: { duration: 0.5, ease: 'easeOut' as const } },
    },
    ...(isInteractive && {
      whileHover: 'hovered',
      whileTap: { scale: 0.97 },
      onClick,
    }),
  }

  return (
    <motion.div
      {...(isInteractive ? { as: 'button' } : {})}
      {...containerProps}
      // render as <button> when interactive, <div> otherwise
      style={isInteractive ? { background: 'none', border: 'none', padding: 0 } : {}}
    >
      {/* ── Icon ── */}
      <motion.div
        className="relative flex-shrink-0"
        style={{ width: iconPx, height: iconPx }}
        variants={isInteractive ? {
          hovered: { scale: 1.08, transition: { duration: 0.2 } },
        } : {}}
      >
        {/* Glow backdrop */}
        <motion.div
          className="absolute inset-0 rounded-xl"
          style={{
            background: 'linear-gradient(135deg, #1d4ed8 0%, #4f46e5 100%)',
            filter: 'blur(0px)',
          }}
          variants={isInteractive ? {
            hovered: {
              filter: 'blur(4px)',
              opacity: 0.7,
              scale: 1.15,
              transition: { duration: 0.3 },
            },
          } : {}}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        />

        {/* Icon SVG — always 40×40 viewBox, scaled via width/height */}
        <svg
          width={iconPx}
          height={iconPx}
          viewBox="0 0 40 40"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="relative z-10"
        >
          <defs>
            <linearGradient id="bgGrad" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stopColor="#1d4ed8" />
              <stop offset="100%" stopColor="#4f46e5" />
            </linearGradient>
            <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#93c5fd" />
              <stop offset="100%" stopColor="#60a5fa" stopOpacity="0.6" />
            </linearGradient>
            <filter id="barGlow">
              <feGaussianBlur stdDeviation="0.8" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Background rounded square */}
          <rect width="40" height="40" rx="9" fill="url(#bgGrad)" />
          {/* Subtle inner border */}
          <rect width="40" height="40" rx="9" fill="none" stroke="white" strokeOpacity="0.12" strokeWidth="1" />

          {/* Animated waveform bars */}
          {BAR_CONFIGS.map((bar, i) => (
            <motion.rect
              key={i}
              x={bar.x}
              width={BAR_WIDTH}
              rx={BAR_WIDTH / 2}
              fill="url(#barGrad)"
              filter="url(#barGlow)"
              initial={{ height: bar.baseH, y: BAR_Y_BASE - bar.baseH }}
              animate={{
                height: bar.animH,
                y: bar.animH.map(h => BAR_Y_BASE - h),
              }}
              transition={{
                duration: bar.duration,
                delay: bar.delay,
                repeat: Infinity,
                repeatType: 'loop',
                ease: 'easeInOut',
              }}
            />
          ))}

          {/* AI spark dot */}
          <motion.circle
            cx="34" cy="7" r="2.5" fill="#a5f3fc"
            animate={{ opacity: [0.5, 1, 0.5], scale: [0.8, 1.2, 0.8] }}
            transition={{ duration: 2.4, repeat: Infinity, ease: 'easeInOut' }}
          />
          <motion.circle
            cx="34" cy="7" r="4" fill="#22d3ee" fillOpacity="0.25"
            animate={{ scale: [1, 1.6, 1], opacity: [0.3, 0, 0.3] }}
            transition={{ duration: 2.4, repeat: Infinity, ease: 'easeInOut' }}
          />
        </svg>
      </motion.div>

      {/* ── Text ── */}
      <div className="flex flex-col leading-tight text-left">
        <motion.span
          className={`font-bold ${titleCls} tracking-tight text-foreground`}
          style={{ letterSpacing: '-0.01em' }}
          variants={isInteractive ? { hovered: { x: 2, transition: { duration: 0.2 } } } : {}}
        >
          Transcript
        </motion.span>
        <motion.span
          className={`${subtitleCls} font-semibold tracking-widest uppercase`}
          style={{
            background: 'linear-gradient(90deg, #3b82f6, #6366f1, #3b82f6)',
            backgroundSize: '200% 100%',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}
          animate={{ backgroundPosition: ['0% 0%', '100% 0%', '0% 0%'] }}
          transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
          variants={isInteractive ? { hovered: { x: 2, transition: { duration: 0.2 } } } : {}}
        >
          Intelligence
        </motion.span>
      </div>
    </motion.div>
  )
}
