import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { RemixControls, REMIX_DEFAULTS } from '../RemixControls'

describe('RemixControls', () => {
  it('renders all three transition bar options', () => {
    render(<RemixControls value={REMIX_DEFAULTS} onChange={() => {}} />)
    expect(screen.getByText('8')).toBeInTheDocument()
    expect(screen.getByText('16')).toBeInTheDocument()
    expect(screen.getByText('32')).toBeInTheDocument()
  })

  it('calls onChange with transition_bars=8 when 8 is clicked', async () => {
    const onChange = vi.fn()
    render(<RemixControls value={REMIX_DEFAULTS} onChange={onChange} />)
    await userEvent.click(screen.getByText('8'))
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ transition_bars: 8 }))
  })

  it('default bridge_beat_mode is none', () => {
    expect(REMIX_DEFAULTS.bridge_beat_mode).toBe('none')
  })
})
