import os
import re
from dataclasses import dataclass


SAFE_CHARS_PATTERN = re.compile(r"[^A-Za-z0-9 ._-]")


@dataclass(frozen=True)
class ImageIdentity:
    original_name: str
    base_name: str
    sample: str | None = None
    slide: str | None = None
    owner: str | None = None
    objective: str | None = None
    well_position: str | None = None
    probe: str | None = None

    @property
    def well_probe_key(self) -> str:
        if self.well_position and self.probe:
            return f"{self.well_position}_{self.probe}"
        return self.base_name


@dataclass(frozen=True)
class HistogramIdentity:
    image: ImageIdentity
    mask_name: str
    source_masks: tuple[str, ...] = ()

    @property
    def mask_base_name(self) -> str:
        return strip_extension(self.mask_name)

    @property
    def display_name(self) -> str:
        return f"{self.image.base_name} under {self.mask_base_name}"

    @property
    def color_key(self) -> str:
        return self.image.well_probe_key

    @property
    def safe_filename_base(self) -> str:
        parts = [
            self.image.sample,
            self.image.slide,
            self.mask_base_name,
            self.image.well_position,
            self.image.probe,
        ]

        if all(parts):
            return sanitize_name("_".join(parts))

        return sanitize_name(f"{self.mask_base_name}_{self.image.base_name}")

    def to_metadata(self, color: str | None = None) -> dict:
        return {
            "image_name": self.image.original_name,
            "mask_name": self.mask_name,
            "source_masks": list(self.source_masks),
            "histogram_name": self.safe_filename_base,
            "well_position": self.image.well_position,
            "probe": self.image.probe,
            "color_key": self.color_key,
            "color": color,
        }


def strip_extension(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def sanitize_name(name: str) -> str:
    return SAFE_CHARS_PATTERN.sub("_", name)


def parse_image_identity(image_name: str) -> ImageIdentity:
    """
    Expected image format:
    <Sample>_<Slide ##>_<Owner Initials>_<ObjectiveMag>_<Well Position>_<Probe>
    """
    base_name = strip_extension(image_name)
    safe_base = sanitize_name(base_name)
    parts = safe_base.split("_")

    if len(parts) >= 6:
        return ImageIdentity(
            original_name=image_name,
            base_name=safe_base,
            sample=parts[0],
            slide=parts[1],
            owner=parts[2],
            objective=parts[3],
            well_position=parts[4],
            probe=parts[5],
        )

    return ImageIdentity(
        original_name=image_name,
        base_name=safe_base,
    )


def build_histogram_identity(image_name: str, mask_name: str, source_masks=None) -> HistogramIdentity:
    return HistogramIdentity(
        image=parse_image_identity(image_name),
        mask_name=mask_name,
        source_masks=tuple(source_masks or ()),
    )
