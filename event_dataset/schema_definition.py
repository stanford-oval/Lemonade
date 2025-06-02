from abc import ABC
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from event_dataset.common_schema import AbstractEvent
from event_dataset.location_utils import (
    country_name_to_code,
    search_location_in_openstreetmap,
)
from log_utils import logger

# From ACLED: https://acleddata.com/knowledge-base/codebook/#acled-events


class WomenTargetedCategory(str, Enum):
    CANDIDATES_FOR_OFFICE = "Women who are running in an election to hold a publicly elected government position"
    POLITICIANS = "Women who currently serve in an elected position in government"
    POLITICAL_PARTY_SUPPORTERS = "political party supporters"
    VOTERS = "Women who are registering to vote or are casting a ballot in an election"
    GOVERNMENT_OFFICIALS = "Women who work for the local, regional, or national government in a non-partisan capacity"
    ACTIVISTS_HRD_SOCIAL_LEADERS = (
        "Women who are activists/human rights defenders/social leaders"
    )
    RELATIVES_OF_TARGETED_GROUPS = "Women who are subject to violence as a result of who they are married to, the daughter of, related to, or are otherwise personally connected to (e.g. candidates, politicians, social leaders, armed actors, voters, party supporters, etc.)"
    ACCUSED_OF_WITCHCRAFT = "Women accused of witchcraft or sorcery, or other mystical or spiritual practices that are typically considered taboo or dangerous within some societies (excluding women who serve as religious leaders in religious structures that are typically not viewed as taboo or dangerous, such as nuns, female priests, or shamans)"
    GIRLS = "Girls who are under the age of 18; they may be specifically referred to by age or explicitly referred to as a child/girl"

    def __repr__(self):
        return f"WomenTargetedCategory.{self.name}"


class Location(BaseModel):
    """
    The most specific location for an event. Normalized to be in English, regardless of the language of the text.
    Locations can be named populated places, geostrategic locations, natural locations, or neighborhoods of larger cities.
    In selected large cities with activity dispersed over many neighborhoods, locations are further specified to predefined subsections within a city. In such cases, City Name – District name (e.g. Mosul – Old City) is recorded in "specific_location". If information about the specific neighborhood/district is not known, the location is recorded at the city level (e.g. Mosul).
    """

    country: str = Field(
        ...,
        description="Name of the country in English. Example: United States",
    )
    address: str = Field(
        ...,
        description="Comma-separated address in order from neighborhood level to village/city, district, county, province, region, and country, if available. Excludes street names, buildings, and other specific landmarks. Example: Mosul, Old City, Nineveh, Nineveh, Iraq",
    )

    def get_normalized_dict(self) -> dict[str, str]:
        country_code = country_name_to_code(self.country)
        address_components = [a.strip() for a in self.address.split(",")]

        geocoded_location = self._try_geocoding(country_code, address_components)

        if not geocoded_location:
            logger.warning(
                f"Could not geocode location after attempts: {address_components}",
            )
            return {"country": self.country}

        return self._filter_address_hierarchy(geocoded_location)

    def normalize_for_evaluation(self, gold: "Location") -> dict[str, str]:
        """If the prediction is correct but has extra specific fields, remove them."""
        country_code = country_name_to_code(self.country)
        address_components = [a.strip() for a in self.address.split(",")]
        gold_normalized = gold.get_normalized_dict()

        for i in range(len(address_components)):
            address = ", ".join(address_components[i:])
            geocoded_location = search_location_in_openstreetmap(country_code, address)

            if geocoded_location:
                filtered_location = self._filter_address_hierarchy(geocoded_location)
                if filtered_location == gold_normalized:
                    return filtered_location

        return self.get_normalized_dict()

    def _try_geocoding(
        self, country_code: str, address_components: list[str]
    ) -> dict[str, str] | None:
        """Try geocoding with progressively shorter addresses."""
        for i in range(len(address_components)):
            address = ", ".join(address_components[i:])
            geocoded_location = search_location_in_openstreetmap(country_code, address)
            if geocoded_location:
                return geocoded_location
        return None

    def _filter_address_hierarchy(
        self, geocoded_location: dict[str, str]
    ) -> dict[str, str]:
        """Filter geocoded location to only include keys in the address hierarchy."""
        return {k: v for k, v in geocoded_location.items() if k in address_hierarchy}

    @staticmethod
    def empty() -> "Location":
        return Location(country="", address="")


class Event(AbstractEvent, ABC):
    """A general class to represent events."""

    location: Location = Field(...)

    def split_to_arguments(
        self,
        include_values: bool = True,
        include_event_type: bool = True,
        gold_location: Optional["Location"] = None,
    ) -> tuple[list[str], list[str], list[str]]:
        # TODO this only supports one level of nesting, which is enough for now
        event_type_string = f"{self.__class__.__name__}." if include_event_type else ""

        location_fields: list[str] = []
        entity_fields: list[str] = []
        other_fields: list[str] = []

        def add_to_results(
            event_type_string: str, field_name: str, field_value, include_values: bool
        ) -> None:
            field_string = f"{event_type_string}{field_name}"
            if include_values:
                field_string += f"={field_value}"

            if "location" in field_name:
                location_fields.append(field_string)
            elif field_name in self.get_entity_field_names():
                entity_fields.append(field_string)
            else:
                other_fields.append(field_string)

        for field_name, field_value in self.__dict__.items():
            if field_name == "crowd_size" or not field_value:
                continue  # Skip crowd_size and empty values

            if isinstance(field_value, (str, int, bool)) or field_value is None:
                add_to_results(
                    event_type_string, field_name, field_value, include_values
                )
            elif isinstance(field_value, list):
                for item in field_value:
                    add_to_results(event_type_string, field_name, item, include_values)
            elif isinstance(field_value, Location):
                location_dict = (
                    field_value.normalize_for_evaluation(gold_location)
                    if gold_location
                    else field_value.get_normalized_dict()
                )
                for k, v in location_dict.items():
                    add_to_results(
                        event_type_string, f"{field_name}.{k}", v, include_values
                    )
            elif isinstance(field_value, BaseModel):
                for nested_field, nested_field_value in field_value.__dict__.items():
                    if nested_field_value:
                        add_to_results(
                            event_type_string,
                            f"{field_name}.{nested_field}",
                            nested_field_value,
                            include_values,
                        )

        return location_fields, entity_fields, other_fields

    def __hash__(self) -> int:
        return hash(repr(self))


address_hierarchy = [
    "country",
    "territory",
    "state",
    "state_district",
    "region",
    "province",
    "county",
    "district",
    "regency",
    "municipality",
    "city",
    "archipelago",
    "island",
    "city_district",
    "subdistrict",
    "town",
    "township",
    "borough",
    "suburb",
    "quarter",
    "village",
    "hamlet",
    "locality",
    "civil_parish",
    "neighborhood",
    "farm",
    "isolated_dwelling",
    "city_block",
]


# Docstring of parent classes is NOT fed to the LLM, only their Fields and Field descriptions
## Battles
class Battle(Event, ABC):
    """
    A "Battle" event is defined as a violent interaction between two organized armed groups at a particular time and location. "Battle" can occur between armed and organized state, non-state, and external groups, and in any combination therein. There is no fatality minimum necessary for inclusion. Civilians can be harmed in the course of larger "Battle" events if they are caught in the crossfire, for example, or affected by strikes on military targets, which is commonly referred to as "collateral damage" (for more, see Indirect Killing of Civilians). When civilians are harmed in a "Battle" event, they are not recorded as an "Actor", nor is a separate civilian-specific event recorded. If any civilian fatalities are reported as part of a battle, they are aggregated in the "Fatalities" field for the "Battle" event.
    The specific elements of the definition of a "Battle" event are as follows:
    Violent interaction: the exchange of armed force, or the use of armed force at close distance, between armed groups capable of inflicting harm upon the opposing side.
    Organized armed groups: collective actors assumed to be operating cohesively around an agenda, identity, or political purpose, using weapons to inflict harm. These groups frequently have a designated name and stated agenda.
    The "Battle" event type may include: ground clashes between different armed groups, ground clashes between armed groups supported by artillery fire or airstrikes, ambushes of on-duty soldiers or armed militants, exchanges of artillery fire, ground attacks against military or militant positions, air attacks where ground forces are able to effectively fire on the aircraft, and air-to-air combat.
    Cases where territory is regained or overtaken without resistance or armed interaction are not recorded as "Battle" events. Instead, they are recorded as "NonStateActorOvertakesTerritory" under the "StrategicDevelopment" event type
    "Battle" event type has the following subtypes:
    - GovernmentRegainsTerritory: Government forces or their affiliates regain control of a location from competing state forces or non-state groups through armed interaction.
    - NonStateActorOvertakesTerritory: A non-state actor or foreign state actor captures territory from an opposing government or non-state actor through armed interaction, establishing a monopoly of force within that territory.
    - ArmedClash: Armed, organized groups engage in a battle without significant changes in territorial control.

    """

    fatalities: Optional[int] = Field(
        None,
        description="Total number of fatalities, if known",
    )


class GovernmentRegainsTerritory(Battle):
    """
    Is a type of "Battle" event. This event type is used when government forces or their affiliates that are fighting against competing state forces or against a non-state group regain control of a location through armed interaction. This event type is only recorded for the re-establishment of government control and not for cases where competing non-state actors exchange control. Short-lived and/or small-scale territorial exchanges that do not last for more than one day are recorded as "ArmedClash".
    """

    government_force: list[str] = Field(
        default=...,
        description="The government forces or their affiliates that regain control of the territory",
        json_schema_extra={"is_entity_field": True},
    )
    adversary: list[str] = Field(
        default=...,
        description="The competing state forces or non-state group that lose control of the territory. Can be State Forces, Rebel Groups, Political Militias, Identity Militias or External Forces",
        json_schema_extra={"is_entity_field": True},
    )


class NonStateActorOvertakesTerritory(Battle):
    """
    Is a type of "Battle" event. This event type is used when a non-state actor (excluding those operating directly on behalf of the government) or a foreign state actor, through armed interaction, captures territory from an opposing government or non-state actor; as a result, they are regarded as having a monopoly of force within that territory. Short-lived and/or small-scale territorial exchanges that do not last for more than one day are recorded as "ArmedClash" events. In cases where non-state forces fight with opposing actors in a location many times before gaining control, only the final territorial acquisition is recorded as "Non-state actor overtakes territory". All other battles in that location are recorded as "ArmedClash".
    """

    non_state_actor: list[str] = Field(
        ...,
        description="The non-state actor overtaking territory. Can be Rebel Groups, Political Militias, Identity Militias or External Forces",
        json_schema_extra={"is_entity_field": True},
    )
    adversary: list[str] = Field(
        ...,
        description="The opposing government or non-state actor from whom the territory was taken. Can be State Forces, Rebel Groups, Political Militias, Identity Militias or External Forces",
        json_schema_extra={"is_entity_field": True},
    )


class ArmedClash(Battle):
    """
    Is a type of "Battle" event. This event type is used when two organized groups like State Forces, Rebel Groups, Political Militias, Identity Militias or External Forces engage in a battle, and no reports indicate a significant change in territorial control.
    `side_1` and `side_2` denote the two sides of the armed clash.
    Excludes demonstrations that turn violent, riots, and other forms of violence that are not organized armed clashes.
    """

    side_1: list[str] = Field(
        ...,
        description="Groups involved in the clash. Can be State Forces, Rebel Groups, Political Militias, Identity Militias or External Forces",
        json_schema_extra={"is_entity_field": True},
    )
    side_2: list[str] = Field(
        ...,
        description="Groups involved in the clash. Can be State Forces, Rebel Groups, Political Militias, Identity Militias or External Forces",
        json_schema_extra={"is_entity_field": True},
    )
    targets_local_administrators: bool = Field(
        ...,
        description="Whether this violent event is affecting current local government officials and administrators — including governors, mayors, councilors, and other civil servants.",
    )
    women_targeted: list[WomenTargetedCategory] = Field(
        ...,
        description="The category of violence against women, if any. If this violence is not targeting women, this should be an empty list.",
    )


### Protests
class Protest(Event, ABC):
    """
    A "Protest" event is defined as an in-person public demonstration of three or more participants in which the participants do not engage in violence, though violence may be used against them. Events include individuals and groups who peacefully demonstrate against a political entity, government institution, policy, group, tradition, business, or other private institution. The following are not recorded as "Protest" events: symbolic public acts such as displays of flags or public prayers (unless they are accompanied by a demonstration); legislative protests, such as parliamentary walkouts or members of parliaments staying silent; strikes (unless they are accompanied by a demonstration); and individual acts such as self-harm actions like individual immolations or hunger strikes.
    Protestor are noted by generic actor name "Protestor". If they are representing a group, the name of that group is also recorded in the field.
    "Protest" event type has the following subtypes:
    - ExcessiveForceAgainstProtestors: Peaceful protestor are targeted with lethal violence or violence resulting in serious injuries by state or non-state actors.
    - ProtestWithIntervention: A peaceful protest is physically dispersed or suppressed without serious injuries, or protestor interact with armed groups or rioters without serious harm, or protestors are arrested.
    - PeacefulProtest: Demonstrators gather for a protest without engaging in violence or rioting and are not met with force or intervention.

    """

    crowd_size: Optional[str] = Field(
        None,
        description="Estimated size of the crowd. It can be an exact number, a range, or a qualitative description like 'small'.",
    )
    protestors: list[str] = Field(
        ...,
        description="List of protestor groups or individuals involved in the protest",
        json_schema_extra={"is_entity_field": True},
    )


class ExcessiveForceAgainstProtestors(Protest):
    """
    Is a type of "Protest" event (Protest events include individuals and groups who peacefully demonstrate against a political entity, government institution, policy, group, tradition, business, or other private institution.) This event type is used when individuals are engaged in a peaceful protest and are targeted with lethal violence or violence resulting in serious injuries (e.g. requiring hospitalization). This includes situations where remote explosives, such as improvised explosive devices, are used to target protestors, as well as situations where non-state actors, such as rebel groups, target protestors.
    """

    perpetrators: list[str] = Field(
        ...,
        description="Entities perpetrating the violence. Can be State Forces, Rebel Groups, Political Militias, Identity Militias, External Forces",
        json_schema_extra={"is_entity_field": True},
    )
    targets_civilians: bool = Field(
        ...,
        description="Indicates if the 'ExcessiveForceAgainstProtestors' event is mainly or only targeting civilians. E.g. state forces using lethal force to disperse peaceful protestors.",
    )

    fatalities: Optional[int] = Field(
        None,
        description="Total number of fatalities, if known",
    )


class ProtestWithIntervention(Protest):
    """
    Is a type of "Protest" event. This event type is used when individuals are engaged in a peaceful protest during which there is a physically violent attempt to disperse or suppress the protest, which resulted in arrests, or minor injuries . If there is intervention, but not violent, the event is recorded as "PeacefulProtest" event type.
    """

    perpetrators: list[str] = Field(
        ...,
        description="Group(s) or entities attempting to disperse or suppress the protest",
        json_schema_extra={"is_entity_field": True},
    )
    fatalities: Optional[int] = Field(
        None,
        description="Total number of fatalities, if known",
    )


class PeacefulProtest(Protest):
    """
    Is a type of "Protest" event (Protest events include individuals and groups who peacefully demonstrate against a political entity, government institution, policy, group, tradition, business, or other private institution.) This event type is used when demonstrators gather for a protest and do not engage in violence or other forms of rioting activity, such as property destruction, and are not met with any sort of violent intervention.
    """

    counter_protestors: list[str] = Field(
        ...,
        description="Groups or entities engaged in counter protest, if any",
        json_schema_extra={"is_entity_field": True},
    )


## Riots
class Riot(Event, ABC):
    """
    "Riot" are violent events where demonstrators or mobs of three or more engage in violent or destructive acts, including but not limited to physical fights, rock throwing, property destruction, etc. They may engage individuals, property, businesses, other rioting groups, or armed actors. Rioters are noted by generic actor name "Rioters". If rioters are affiliated with a specific group – which may or may not be armed – or identity group, that group is recorded in the respective "Actor" field. Riots may begin as peaceful protests, or a mob may have the intention to engage in violence from the outset.
    "Riot" event type has the following subtypes:
    - ViolentDemonstration: Demonstrators engage in violence or destructive activities, such as physical clashes, vandalism, or road-blocking, regardless of who initiated the violence.
    - MobViolence: Rioters violently interact with other rioters, civilians, property, or armed groups outside of demonstration contexts, often involving disorderly crowds with the intention to cause harm or disruption.

    """

    crowd_size: Optional[str] = Field(
        None,
        description="Estimated size of the crowd. It can be an exact number, a range, or a qualitative description like 'small'.",
    )
    fatalities: Optional[int] = Field(
        None,
        description="Total number of fatalities, if known",
    )
    targets_civilians: bool = Field(
        ...,
        description="Indicates if the 'Riot' event is mainly or only targeting civilians. E.g. a village mob assaulting another villager over a land dispute.",
    )
    group_1: list[str] = Field(
        ...,
        description="Group or individual involved in the violence",
        json_schema_extra={"is_entity_field": True},
    )
    group_2: list[str] = Field(
        ...,
        description="The other group or individual involved in the violence, if any",
        json_schema_extra={"is_entity_field": True},
    )
    targets_local_administrators: bool = Field(
        ...,
        description="Whether this violent event is affecting current local government officials and administrators — including governors, mayors, councilors, and other civil servants.",
    )
    women_targeted: list[WomenTargetedCategory] = Field(
        ...,
        description="The category of violence against women, if any. If this violence is not targeting women, this should be an empty list.",
    )


class ViolentDemonstration(Riot):
    """
    Is a type of "Riot" event. This event type is used when demonstrators engage in violence and/or destructive activity. Examples include physical clashes with other demonstrators or government forces; vandalism; and road-blocking using barricades, burning tires, or other material. The coding of an event as a "Violent demonstration" does not necessarily indicate that demonstrators initiated the violence and/or destructive actions.
    Excludes events where a weapon is drawn but not used, or when the situation is de-escalated before violence occurs.
    """


class MobViolence(Riot):
    """
    Is a type of "Riot" event. A mob is considered a crowd of people that is disorderly and has the intention to cause harm or disruption through violence or property destruction. Note that this type of violence can also include spontaneous vigilante mobs clashing with other armed groups or attacking civilians. While a "Mob violence" event often involves unarmed or crudely armed rioters, on rare occasions, it can involve violence by people associated with organized groups and/or using more sophisticated weapons, such as firearms.
    """


## Explosions/Remote violence
class ExplosionOrRemoteViolence(Event, ABC):
    """
    "ExplosionOrRemoteViolence" is defined as events as incidents in which one side uses weapon types that, by their nature, are at range and widely destructive. The weapons used in "ExplosionOrRemoteViolence" events are explosive devices, including but not limited to: bombs, grenades, improvised explosive devices (IEDs), artillery fire or shelling, missile attacks, air or drone strikes, and other widely destructive heavy weapons or chemical weapons. Suicide attacks using explosives also fall under this category. When an "ExplosionOrRemoteViolence" event is reported in the context of an ongoing battle, it is merged and recorded as a single "Battles" event. "ExplosionOrRemoteViolence" can be used against armed agents as well as civilians.
    "ExplosionOrRemoteViolence" event type has the following subtypes:
    - ChemicalWeapon: The use of chemical weapons in warfare without any other engagement.
    - AirOrDroneStrike: Air or drone strikes occurring without any other engagement, including attacks by helicopters.
    - SuicideBomb: A suicide bombing or suicide vehicle-borne improvised explosive device (SVBIED) attack without an armed clash.
    - ShellingOrArtilleryOrMissileAttack: The use of long-range artillery, missile systems, or other heavy weapons platforms without any other engagement.
    - RemoteExplosiveOrLandmineOrIED: Detonation of remotely- or victim-activated devices, including landmines and IEDs, without any other engagement.
    - Grenade: The use of a grenade or similar hand-thrown explosive without any other engagement.
    """

    targets_civilians: bool = Field(
        ...,
        description="Indicates if the event is mainly or only targeting civilians. E.g. a landmine killing a farmer.",
    )
    fatalities: Optional[int] = Field(
        None,
        description="Total number of fatalities, if known",
    )
    attackers: list[str] = Field(
        ...,
        description="Entities conducting the violence",
        json_schema_extra={"is_entity_field": True},
    )
    targeted_entities: list[str] = Field(
        ...,
        description="Entities or actors being targeted",
        json_schema_extra={"is_entity_field": True},
    )
    targets_local_administrators: bool = Field(
        ...,
        description="Whether this violent event is affecting current local government officials and administrators — including governors, mayors, councilors, and other civil servants.",
    )
    women_targeted: list[WomenTargetedCategory] = Field(
        ...,
        description="The category of violence against women, if any. If this violence is not targeting women, this should be an empty list.",
    )


class ChemicalWeapon(ExplosionOrRemoteViolence):
    """
    Is a type of "ExplosionOrRemoteViolence" event. This event type captures the use of chemical weapons in warfare in the absence of any other engagement. Chemical weapons are all substances listed as Schedule 1 of the Chemical Weapons Convention, including sarin gas, mustard gas, chlorine gas, and anthrax. Napalm and white phosphorus, as well as less-lethal crowd control substances – such as tear gas – are not considered chemical weapons within this event type.
    """


class AirOrDroneStrike(ExplosionOrRemoteViolence):
    """
    Is a type of "ExplosionOrRemoteViolence" event. This event type is used when air or drone strikes take place in the absence of any other engagement. Please note that any air-to-ground attacks fall under this event type, including attacks by helicopters that do not involve exchanges of fire with forces on the ground.
    """


class SuicideBomb(ExplosionOrRemoteViolence):
    """
    Is a type of "ExplosionOrRemoteViolence" event. This event type is used when a suicide bombing occurs in the absence of an armed clash, such as an exchange of small arms fire with other armed groups. It also includes suicide vehicle-borne improvised explosive device (SVBIED) attacks. Note that the suicide bomber is included in the total number of reported fatalities coded for such events.
    """


class ShellingOrArtilleryOrMissileAttack(ExplosionOrRemoteViolence):
    """
    Is a type of "ExplosionOrRemoteViolence" event. This event type captures the use of long-range artillery, missile systems, or other heavy weapons platforms in the absence of any other engagement. When two armed groups exchange long-range fire, it is recorded as an "ArmedClash". "ShellingOrArtilleryOrMissileAttack" events include attacks described as shelling, the use of artillery and cannons, mortars, guided missiles, rockets, grenade launchers, and other heavy weapons platforms. Crewed aircraft shot down by long-range systems fall under this event type.  Uncrewed armed drones that are shot down, however, are recorded as interceptions under "DisruptedWeaponsUse" because people are not targeted (see below). Similarly, an interception of a missile strike itself (such as by the Iron Dome in Israel) is also recorded as "DisruptedWeaponsUse".
    """


class RemoteExplosiveOrLandmineOrIED(ExplosionOrRemoteViolence):
    """
    Is a type of "ExplosionOrRemoteViolence" event. This event type is used when remotely- or victim-activated devices are detonated in the absence of any other engagement. Examples include landmines, IEDs – whether alone or attached to a vehicle, or any other sort of remotely detonated or triggered explosive. Unexploded ordnances (UXO) also fall under this category.
    SVBIEDs are recorded as "Suicide bomb" events, while the safe defusal of an explosive or its accidental detonation by the actor who planted it (with no other casualties reported) is recorded under "DisruptedWeaponsUse".
    """


class Grenade(ExplosionOrRemoteViolence):
    """
    Is a type of "ExplosionOrRemoteViolence" event. This event type captures the use of a grenade or any other similarly hand-thrown explosive, such as an IED that is thrown, in the absence of any other engagement. Events involving so-called "crude bombs" (such as Molotov cocktails, firecrackers, cherry bombs, petrol bombs, etc.) as well as "stun grenades" are not recorded in this category, but are included under either "Riot" or "StrategicDevelopment" depending on the context in which they occurred.
    """


## Violence against civilians


class ViolenceAgainstCivilians(Event, ABC):
    """
    "ViolenceAgainstCivilians" are violent events where an organized armed group inflicts violence upon unarmed non-combatants. By definition, civilians are unarmed and cannot engage in political violence. Therefore, the violence is understood to be asymmetric as the perpetrator is assumed to be the only actor capable of using violence in the event. The perpetrators of such acts include state forces and their affiliates, rebels, militias, and external/other forces.
    In cases where the identity and actions of the targets are in question (e.g. the target may be employed as a police officer), it is determined that if a person is harmed or killed while unarmed and unable to either act defensively or counter-attack, this is an act of "ViolenceAgainstCivilians". This includes extrajudicial killings of detained combatants or unarmed prisoners of war.
    "ViolenceAgainstCivilians" also includes attempts at inflicting harm (e.g. beating, shooting, torture, rape, mutilation, etc.) or forcibly disappearing (e.g. kidnapping and disappearances) civilian actors. Note that the "ViolenceAgainstCivilians" event type exclusively captures violence targeting civilians that does not occur concurrently with other forms of violence – such as rioting – that are coded higher in the event type hierarchy. To get a full list of events in the dataset where civilians were the main or only target of violence, users can filter on the "Civilian targeting" field.
    "ViolenceAgainstCivilians" event type has the following subtypes:
    - SexualViolence: Any event where an individual is targeted with sexual violence, including but not limited to rape, public stripping, and sexual torture, with the gender identities of victims recorded when reported.
    - Attack: An event where civilians are targeted with violence by an organized armed actor outside the context of other forms of violence, including severe government overreach by law enforcement.
    - AbductionOrForcedDisappearance: An event involving the abduction or forced disappearance of civilians without reports of further violence, including arrests by non-state groups and extrajudicial detentions by state forces, but excluding standard judicial arrests by state forces.
    """

    targets_local_administrators: bool = Field(
        ...,
        description="Whether this violent event is affecting current local government officials and administrators — including governors, mayors, councilors, and other civil servants.",
    )
    women_targeted: list[WomenTargetedCategory] = Field(
        ...,
        description="The category of violence against women, if any. If this violence is not targeting women, this should be an empty list.",
    )


class SexualViolence(ViolenceAgainstCivilians):
    """
    Is a type of "ViolenceAgainstCivilians" event. This event type is used when any individual is targeted with sexual violence. SexualViolence is defined largely as an action that inflicts harm of a sexual nature. This means that it is not limited to solely penetrative rape, but also includes actions like public stripping, sexual torture, etc. Given the gendered nature of sexual violence, the gender identities of the victims – i.e. "Women", "Men", and "LGBTQ+", or a combination thereof – are recorded in the "Associated Actor" field for these events when reported. Note that it is possible for sexual violence to occur within other event types such as "Battle" and "Riot".
    """

    fatalities: Optional[int] = Field(
        None,
        description="Total number of fatalities, if known",
    )  # Is very very rare, only 7 events in English for 2024
    perpetrators: list[str] = Field(
        ...,
        description="The attacker(s) entity or actor",
        json_schema_extra={"is_entity_field": True},
    )
    victims: list[str] = Field(
        ...,
        description="The entity or actor(s) that is the target or victim of the SexualViolence event",
        json_schema_extra={"is_entity_field": True},
    )


class Attack(ViolenceAgainstCivilians):
    """
    Is a type of "ViolenceAgainstCivilians" event. This event type is used when civilians are targeted with violence by an organized armed actor outside the context of other forms of violence like ArmedClash, Protests, Riots, or ExplosionOrRemoteViolence. Violence by law enforcement that constitutes severe government overreach is also recorded as an "Attack" event.
    Attacks of a sexual nature are recorded as SexualViolence.
    If only property is attacked and not people, the event should be recorded as LootingOrPropertyDestruction event type.
    Excludes discovery of mass graves, which are recorded as "OtherStrategicDevelopment" events.
    """

    fatalities: Optional[int] = Field(
        ...,
        description="Total number of fatalities, if known",
    )
    attackers: list[str] = Field(
        ...,
        description="The attacker entity or actor(s)",
        json_schema_extra={"is_entity_field": True},
    )
    targeted_entities: list[str] = Field(
        ...,
        description="The entity or actor(s) that is the target of the attack",
        json_schema_extra={"is_entity_field": True},
    )


class AbductionOrForcedDisappearance(ViolenceAgainstCivilians):
    """
    Is a type of "ViolenceAgainstCivilians" event. This event type is used when an actor engages in the abduction or forced disappearance of civilians, without reports of further violence. If fatalities or serious injuries are reported during the abduction or forced disappearance, the event is recorded as an "Attack" event instead. If such violence is reported in later periods during captivity, this is recorded as an additional "Attack" event. Note that multiple people can be abducted in a single "Abduction/forced disappearance" event.
    Arrests by non-state groups and extrajudicial detentions by state forces are considered "Abduction/forced disappearance". Arrests conducted by state forces within the standard judicial process are, however, considered "Arrest".
    """

    abductor: list[str] = Field(
        ...,
        description="The abductor person or group(s)",
        json_schema_extra={"is_entity_field": True},
    )
    abductee: list[str] = Field(
        ...,
        description="People or group(s) that were abducted or disappeared. Note that multiple people can be abducted in a single AbductionOrForcedDisappearance event",
        json_schema_extra={"is_entity_field": True},
    )


## Strategic developments
# "strategic developments" should not be assumed to be cross-context and -time comparable as other ACLED event types can be.
# See https://acleddata.com/acleddatanew/wp-content/uploads/2021/11/ACLED_Using-Strategic-Developments_v1_April-2019.pdf


class StrategicDevelopment(Event, ABC):
    """
    This event type captures contextually important information regarding incidents and activities of groups that are not recorded as "Political violence" or "Demonstration" events, yet may trigger future events or contribute to political dynamics within and across states. The inclusion of such events is limited, as their purpose is to capture pivotal events within the broader political landscape. They typically include a disparate range of events, such as recruitment drives, looting, and incursions, as well as the location and date of peace talks and the arrests of high-ranking officials or large groups. While it is rare for fatalities to be reported as a result of such events, they can occur in certain cases – e.g. the suspicious death of a high-ranking official, the accidental detonation of a bomb resulting in the bomber being killed, etc.
    Due to their context-specific nature, "StrategicDevelopment" are not collected and recorded in the same cross-comparable fashion as "Political violence" and "Demonstration" events. As such, the "StrategicDevelopment" event type is primarily a tool for understanding particular contexts.
    "StrategicDevelopment" event type has the following subtypes:
    - Agreement: Records any agreement between different actors, such as peace talks, ceasefires, or prisoner exchanges.
    - Arrest: Used when state forces or controlling actors detain a significant individual or conduct politically important mass arrests.
    - ChangeToArmedGroup: Records significant changes in the activity or structure of armed groups, including creation, recruitment, movement, or absorption of forces.
    - DisruptedWeaponsUse: Captures instances where an explosion or remote violence event is prevented, or when significant weapons caches are seized.
    - BaseEstablished: Used when an organized armed group establishes a permanent or semi-permanent base or headquarters.
    - LootingOrPropertyDestruction: Records incidents of looting or seizing goods/property outside the context of other forms of violence or destruction.
    - NonViolentTransferOfTerritory: Used when actors acquire control of a location without engaging in violent interaction with another group.
    - OtherStrategicDevelopment: Covers significant developments that don't fall into other Strategic Development event types, such as coups or population displacements.
    """


class Agreement(StrategicDevelopment):
    """
    Is a type of "StrategicDevelopment" event. This event type is used to record any sort of agreement between different armed actors (such as governments and rebel groups). Examples include peace agreements/talks, ceasefires, evacuation deals, prisoner exchanges, negotiated territorial transfers, prisoner releases, surrenders, repatriations, etc.
    Excludes agreements between political parties, trade unions, or other non-armed actors like protestors.
    """

    group_1: list[str] = Field(
        ...,
        description="Group or individual involved in the agreement",
        json_schema_extra={"is_entity_field": True},
    )
    group_2: list[str] = Field(
        ...,
        description="The other group or individual involved in the agreement",
        json_schema_extra={"is_entity_field": True},
    )


class Arrest(StrategicDevelopment):
    """
    Is a type of "StrategicDevelopment" event. This event type is used when state forces or other actors exercising de facto control over a territory either detain a particularly significant individual or engage in politically significant mass arrests. This excludes arrests of individuals for common crimes, such as theft or assault, unless the individual is a high-ranking official or the arrest is politically significant.
    """

    detainers: list[str] = Field(
        ...,
        description="The person or group(s) who detains or jails the detainee(s)",
        json_schema_extra={"is_entity_field": True},
    )
    detainees: list[str] = Field(
        ...,
        description="The person or group(s) being detained or jailed",
        json_schema_extra={"is_entity_field": True},
    )


class ChangeToArmedGroup(StrategicDevelopment):
    """
    Is a type of "StrategicDevelopment" event. This event type is used to record significant changes in the activity or structure of armed groups. It can cover anything from the creation of a new rebel group or a paramilitary wing of the security forces, "voluntary" recruitment drives, movement of forces, or any other non-violent security measures enacted by armed actors. This event type can also be used if one armed group is absorbed into a different armed group or to track large-scale defections.
    """

    armed_group: list[str] = Field(
        ...,
        description="The name of armed group that underwent change",
        json_schema_extra={"is_entity_field": True},
    )
    other_actors: list[str] = Field(
        ...,
        description="Other actors or groups involved. E.g. the government that ordered a change to its army.",
        json_schema_extra={"is_entity_field": True},
    )


class DisruptedWeaponsUse(StrategicDevelopment):
    """
    Is a type of "StrategicDevelopment" event. This event type is used to capture all instances in which an event of "ExplosionOrRemoteViolence" is prevented from occurring, or when armed actors seize significant caches of weapons. It includes the safe defusal of an explosive, the accidental detonation of explosives by those allegedly responsible for planting it, the interception of explosives in the air, as well as the seizure of weapons or weapons platforms such as jets, helicopters, tanks, etc. Note that in cases where a group other than the one that planted an explosive is attempting to render an explosive harmless and it goes off, this is recorded under the "ExplosionOrRemoteViolence" event type, as the explosive has harmed an actor other than the one that planted it.
    """

    attackers: list[str] = Field(
        ...,
        description="The entity or actor(s) responsible for the remote violence",
        json_schema_extra={"is_entity_field": True},
    )
    disruptors: list[str] = Field(
        ...,
        description="The entity or actor(s) disrupting the explosion or remote violence",
        json_schema_extra={"is_entity_field": True},
    )
    targets_local_administrators: bool = Field(
        ...,
        description="Whether this violent event is affecting current local government officials and administrators — including governors, mayors, councilors, and other civil servants.",
    )
    women_targeted: list[WomenTargetedCategory] = Field(
        ...,
        description="The category of violence against women, if any. If this violence is not targeting women, this should be an empty list.",
    )


class BaseEstablished(StrategicDevelopment):
    """
    Is a type of "StrategicDevelopment" event. This event type is used when an organized armed group establishes a permanent or semi-permanent base or headquarters. There are few cases where opposition groups other than rebels can also establish a headquarters or base (e.g. AMISOM forces in Somalia).
    """

    group: list[str] = Field(
        ...,
        description="Entity or group(s) establishing the base",
        json_schema_extra={"is_entity_field": True},
    )


class LootingOrPropertyDestruction(StrategicDevelopment):
    """
    Is a type of "StrategicDevelopment" event. This event type is used when actors engage in looting or seizing goods or property outside the context of other forms of violence or destruction, such as rioting or armed clashes. This excludes the seizure or destruction of weapons or weapons systems, which are captured under the "DisruptedWeaponsUse" event type. This can occur during raiding or after the capture of villages or other populated places by armed groups that occur without reported violence.
    """

    perpetrators: list[str] = Field(
        ...,
        description="The group or entity that does the looting or seizure",
        json_schema_extra={"is_entity_field": True},
    )
    victims: list[str] = Field(
        ...,
        description="The group or entity that was the target of looting or seizure",
        json_schema_extra={"is_entity_field": True},
    )
    targets_local_administrators: bool = Field(
        ...,
        description="Whether this violent event is affecting current local government officials and administrators — including governors, mayors, councilors, and other civil servants.",
    )
    women_targeted: list[WomenTargetedCategory] = Field(
        ...,
        description="The category of violence against women, if any. If this violence is not targeting women, this should be an empty list.",
    )


class NonViolentTransferOfTerritory(StrategicDevelopment):
    """
    Is a type of "StrategicDevelopment" event. This event type is used in situations in which rebels, governments, or their affiliates acquire control of a location without engaging in a violent interaction with another group. Rebels establishing control of a location without any resistance is an example of this event.
    """

    actors_taking_over: list[str] = Field(
        default=...,
        description="The entity or actor(s) establishing control.",
        json_schema_extra={"is_entity_field": True},
    )
    actors_giving_up: list[str] = Field(
        default=...,
        description="The entity or actor(s) giving up territory, if known.",
        json_schema_extra={"is_entity_field": True},
    )


class OtherStrategicDevelopment(StrategicDevelopment):
    """
    Is a type of "StrategicDevelopment" event. This event type is used to cover any significant development that does not fall into any of the other "StrategicDevelopment" event types. Includes the occurrence of a coup, the displacement of a civilian population as a result of fighting, and the discovery of mass graves.
    """

    group_1: list[str] = Field(
        default=...,
        description="Group or individual involved in the StrategicDevelopment",
        json_schema_extra={"is_entity_field": True},
    )
    group_2: list[str] = Field(
        default=...,
        description="The other group or individual involved in the violence, if any",
        json_schema_extra={"is_entity_field": True},
    )
