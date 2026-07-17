/**
 * Image Comparison Slider
 * Supports 3 images with 2 draggable sliders
 */

class ImageComparisonSlider {
  constructor(containerOrId) {
    this.container = (typeof containerOrId === 'string')
        ? document.getElementById(containerOrId)
        : containerOrId;
    if (!this.container) return;
    // Make the instance reachable from the DOM so peers (e.g. carousel clones)
    // can sync state.
    this.container.__sliderInstance = this;

    this.slider = null;
    this.handle1 = null;  // Left handle
    this.handle2 = null;  // Right handle
    this.divider1 = null;
    this.divider2 = null;

    // Two slider positions as percentages (0.0-1.0)
    this.slider1Position = 0.33;  // Initial: 33%
    this.slider2Position = 0.66;  // Initial: 66%

    this.activeHandle = null;  // Currently dragged handle (1 or 2)
    this.isDragging = false;

    this.init();
  }

  init() {
    this.slider = this.container.querySelector('.comparison-slider');
    this.handle1 = this.container.querySelector('.comparison-slider-handle-1');
    this.handle2 = this.container.querySelector('.comparison-slider-handle-2');
    this.divider1 = this.container.querySelector('.comparison-slider-divider-1');
    this.divider2 = this.container.querySelector('.comparison-slider-divider-2');

    if (!this.slider || !this.handle1) return;

    // 2-image mode: no second handle/divider/depth layer
    this.twoImageMode = !this.handle2;
    if (this.twoImageMode) {
      this.slider1Position = 0.5;
      this.slider2Position = 1;
    }

    // Attach events to handles
    this.attachHandleEvents(this.handle1, 1);
    if (this.handle2) this.attachHandleEvents(this.handle2, 2);

    // Click on slider to move nearest handle
    this.slider.addEventListener('click', (e) => this.handleClick(e));

    // Set initial positions
    this.updateAllLayers();
  }

  attachHandleEvents(handle, handleId) {
    handle.addEventListener('mousedown', (e) => this.startDrag(e, handleId));
    handle.addEventListener('touchstart', (e) => this.startDrag(e, handleId));
  }

  startDrag(e, handleId) {
    this.isDragging = true;
    this.activeHandle = handleId;

    const handle = handleId === 1 ? this.handle1 : this.handle2;
    handle.style.cursor = 'grabbing';

    // Prevent text/image dragging behavior
    document.body.style.userSelect = 'none';
    document.body.style.touchAction = 'none';

    // Attach global move/end listeners
    document.addEventListener('mousemove', this.dragHandler);
    document.addEventListener('mouseup', this.endDragHandler);
    document.addEventListener('touchmove', this.dragHandler);
    document.addEventListener('touchend', this.endDragHandler);
  }

  // Use arrow functions to preserve 'this' context
  dragHandler = (e) => this.drag(e);
  endDragHandler = () => this.endDrag();

  drag(e) {
    if (!this.isDragging || this.activeHandle === null) return;

    const rect = this.slider.getBoundingClientRect();
    let clientX;

    // Support touch and mouse
    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
    } else {
      clientX = e.clientX;
    }

    let x = clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));

    let newPosition = x / rect.width;

    // Apply constraints based on which handle is being dragged
    if (this.activeHandle === 1) {
      // Left handle: cannot exceed right handle
      newPosition = Math.min(newPosition, this.slider2Position);
      this.slider1Position = newPosition;
    } else if (this.activeHandle === 2) {
      // Right handle: cannot go below left handle
      newPosition = Math.max(newPosition, this.slider1Position);
      this.slider2Position = newPosition;
    }

    this.updateAllLayers();
    this._notifyChange();
  }

  endDrag() {
    this.isDragging = false;

    if (this.activeHandle === 1) {
      this.handle1.style.cursor = 'grab';
    } else if (this.activeHandle === 2) {
      this.handle2.style.cursor = 'grab';
    }

    this.activeHandle = null;

    document.body.style.userSelect = '';
    document.body.style.touchAction = '';

    // Remove global listeners
    document.removeEventListener('mousemove', this.dragHandler);
    document.removeEventListener('mouseup', this.endDragHandler);
    document.removeEventListener('touchmove', this.dragHandler);
    document.removeEventListener('touchend', this.endDragHandler);
  }

  handleClick(e) {
    // Prevent click from triggering if we just finished dragging
    if (this.isDragging) return;

    // Don't handle clicks on handles themselves
    if (e.target.closest('.comparison-slider-handle')) return;

    const rect = this.slider.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickPosition = clickX / rect.width;

    if (this.twoImageMode) {
      this.slider1Position = clickPosition;
      this.updateAllLayers();
      this._notifyChange();
      return;
    }

    // Determine which handle is closer to click position
    const dist1 = Math.abs(clickPosition - this.slider1Position);
    const dist2 = Math.abs(clickPosition - this.slider2Position);

    if (dist1 < dist2) {
      // Move left handle (with constraint)
      this.slider1Position = Math.min(clickPosition, this.slider2Position);
    } else {
      // Move right handle (with constraint)
      this.slider2Position = Math.max(clickPosition, this.slider1Position);
    }

    this.updateAllLayers();
    this._notifyChange();
  }

  updateAllLayers() {
    const topLayer = this.slider.querySelector('.comparison-slider-top');
    const middleLayer = this.slider.querySelector('.comparison-slider-middle');
    const depthLayer = this.slider.querySelector('.comparison-slider-depth');

    // Top layer: show from 0 to slider1Position
    if (topLayer) {
      const rightInset1 = (1 - this.slider1Position) * 100;
      topLayer.style.clipPath = `inset(0 ${rightInset1}% 0 0)`;
      topLayer.style.webkitClipPath = `inset(0 ${rightInset1}% 0 0)`;
    }

    // Middle layer: show from slider1Position to slider2Position
    if (middleLayer) {
      const leftInset = this.slider1Position * 100;
      const rightInset2 = (1 - this.slider2Position) * 100;
      middleLayer.style.clipPath = `inset(0 ${rightInset2}% 0 ${leftInset}%)`;
      middleLayer.style.webkitClipPath = `inset(0 ${rightInset2}% 0 ${leftInset}%)`;
    }

    // Depth layer: show from slider2Position to 100%
    if (depthLayer) {
      const leftInset2 = this.slider2Position * 100;
      depthLayer.style.clipPath = `inset(0 0 0 ${leftInset2}%)`;
      depthLayer.style.webkitClipPath = `inset(0 0 0 ${leftInset2}%)`;
    }

    // Update handle and divider positions
    this.updateHandlePosition(this.handle1, this.divider1, this.slider1Position);
    this.updateHandlePosition(this.handle2, this.divider2, this.slider2Position);

    // Update label positions to center them in their respective regions
    this.updateLabelPositions();
  }

  updateHandlePosition(handle, divider, position) {
    const leftPercent = position * 100;

    if (handle) {
      handle.style.left = leftPercent + '%';
      handle.style.transform = 'translate(-50%, -50%)';
    }

    if (divider) {
      divider.style.left = leftPercent + '%';
      divider.style.transform = 'translateX(-50%)';
    }
  }

  // Programmatically set positions without dispatching the change event.
  // Used by carousel sync to mirror state across cloned slider instances.
  setPosition(p1, p2) {
    if (typeof p1 === 'number') this.slider1Position = p1;
    if (typeof p2 === 'number' && this.handle2) this.slider2Position = p2;
    this.updateAllLayers();
  }

  _notifyChange() {
    this.container.dispatchEvent(new CustomEvent('slider:position', {
      detail: {
        slider1Position: this.slider1Position,
        slider2Position: this.slider2Position,
      },
      bubbles: true,
    }));
  }

  updateLabelPositions() {
    const leftLabel = this.slider.querySelector('.comparison-label-left');
    const middleLabel = this.slider.querySelector('.comparison-label-middle');
    const rightLabel = this.slider.querySelector('.comparison-label-right');

    // Position labels in the center of each visible region
    if (leftLabel) {
      // Left label: center of region from 0% to slider1Position
      const centerPos = (this.slider1Position / 2) * 100;
      leftLabel.style.left = centerPos + '%';
      leftLabel.style.transform = 'translateX(-50%)';
      leftLabel.style.right = 'auto';
    }

    if (middleLabel) {
      // Middle label: center of region from slider1Position to slider2Position
      const centerPos = ((this.slider1Position + this.slider2Position) / 2) * 100;
      middleLabel.style.left = centerPos + '%';
      middleLabel.style.transform = 'translateX(-50%)';
    }

    if (rightLabel) {
      // Right label: center of region from slider2Position (or slider1Position in 2-image mode) to 100%
      const rightStart = this.twoImageMode ? this.slider1Position : this.slider2Position;
      const centerPos = ((rightStart + 1) / 2) * 100;
      rightLabel.style.left = centerPos + '%';
      rightLabel.style.transform = 'translateX(-50%)';
      rightLabel.style.right = 'auto';
    }
  }
}

// Initialize sliders when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  // Initialize all image comparison sliders
  const sliders = document.querySelectorAll('[data-comparison-slider]');
  sliders.forEach(sliderElement => {
    const id = sliderElement.id;
    if (id) {
      new ImageComparisonSlider(id);
    }
  });
});
