import argostranslate.package
import argostranslate.translate
from itertools import combinations

language_codes: set[str] = {
    "en",
    "ru",
    "es",
    "fr",
}

pairs: set[tuple] = set()

for L in range(len(language_codes) + 1):
    for subset in combinations(language_codes, 2):
        pairs.add(tuple(subset))
        pairs.add(tuple(subset[::-1]))

argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()

for pair in pairs:
    package_to_install = [
        filter(
            lambda x: x.from_code == pair[0] and x.to_code == pair[1],
            available_packages,
        )
    ]
    argostranslate.package.install_from_path(package_to_install.download())
